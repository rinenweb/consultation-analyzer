import re
import time
import unicodedata
from datetime import datetime
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from rapidfuzz import fuzz


def extract_base_and_parent(url_text: str):
    """
    Accepts full URL like:
    https://www.opengov.gr/immigration/?p=2000
    Returns:
      BASE = https://www.opengov.gr/immigration/
      parent_id = 2000
    """
    if not url_text:
        return None, None

    m = re.search(r"^https?://www\.opengov\.gr/([^/]+)/\?p=(\d+)", url_text.strip())
    if not m:
        return None, None

    ministry = m.group(1)
    parent_id = m.group(2)
    base = f"https://www.opengov.gr/{ministry}/"
    return base, parent_id


def build_comment_link(base: str, comment_id: str) -> str:
    return f"{base}?c={comment_id}"

GREEK_MONTHS = {
    "ιανουαριου": 1,
    "φεβρουαριου": 2,
    "μαρτιου": 3,
    "απριλιου": 4,
    "μαιου": 5,
    "ιουνιου": 6,
    "ιουλιου": 7,
    "αυγουστου": 8,
    "σεπτεμβριου": 9,
    "οκτωβριου": 10,
    "νοεμβριου": 11,
    "δεκεμβριου": 12,
}


def parse_greek_datetime(dt_text: str):
    """
    Parses strings like:
    '2 Μαρτίου 2026, 15:30'
    Returns datetime or None.
    """
    if not dt_text:
        return None

    clean = canonicalize_text(dt_text).replace(",", "")
    m = re.match(r"(\d{1,2})\s+([^\s]+)\s+(\d{4})\s+(\d{1,2}):(\d{2})", clean)
    if not m:
        return None

    day = int(m.group(1))
    month_name = m.group(2).strip()
    year = int(m.group(3))
    hour = int(m.group(4))
    minute = int(m.group(5))

    month = GREEK_MONTHS.get(month_name)
    if not month:
        return None

    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


def scrape_consultation_timing(parent_id: str, base: str, session: requests.Session, translations: dict):
    """
    Extract posting and closing datetime from the consultation sidebar.
    Returns dict with raw strings, parsed datetimes, duration_days, and qualitative status.
    """
    r = session.get(f"{base}?p={parent_id}", timeout=25)
    soup = BeautifulSoup(r.text, "html5lib")

    timing_box = soup.find("div", class_="sidespot red_spot")

    posted_raw = None
    closes_raw = None
    posted_dt = None
    closes_dt = None
    duration_days = None
    duration_label = None
    duration_color = None

    if timing_box:
        spans = timing_box.find_all("span")
        if len(spans) >= 2:
            posted_raw = spans[0].get_text(" ", strip=True)
            closes_raw = spans[1].get_text(" ", strip=True)

            posted_dt = parse_greek_datetime(posted_raw)
            closes_dt = parse_greek_datetime(closes_raw)

            if posted_dt and closes_dt:
                duration_days = round((closes_dt - posted_dt).total_seconds() / 86400, 2)

                if duration_days < 14:
                    duration_label = translations.get("duration_insufficient", "Insufficient duration")
                    duration_color = "red"
                elif duration_days < 21:
                    duration_label = translations.get("duration_borderline", "Borderline duration")
                    duration_color = "orange"
                else:
                    duration_label = translations.get("duration_satisfactory", "Satisfactory duration")
                    duration_color = "green"

    return {
        "posted_raw": posted_raw,
        "closes_raw": closes_raw,
        "posted_dt": posted_dt.isoformat() if posted_dt else None,
        "closes_dt": closes_dt.isoformat() if closes_dt else None,
        "duration_days": duration_days,
        "duration_label": duration_label,
        "duration_color": duration_color,
    }
# =========================================================
# STEP UI (NO DUPLICATION)
# =========================================================

def render_steps(placeholder, steps, active_idx, done_set):
    """
    steps: list[str]
    active_idx: int (current step index)
    done_set: set[int]
    """
    lines = []
    for i, label in enumerate(steps):
        if i in done_set:
            lines.append(f"✅ {label}")
        elif i == active_idx:
            lines.append(f"⏳ {label}")
        else:
            lines.append(f"▫️ {label}")
    placeholder.markdown("\n\n".join(lines))

# =========================================================
# TEXT CANONICALIZATION (FOR DUPLICATE DETECTION)
# =========================================================

def canonicalize_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================================================
# PURE SCRAPING (WITH PROGRESS + ABORT CHECKS)
# =========================================================

def get_chapters(parent_id: str, base: str, session: requests.Session):
    r = session.get(f"{base}?p={parent_id}", timeout=25)
    soup = BeautifulSoup(r.text, "html5lib")
    nav = soup.find("ul", class_="other_posts")
    chapters = []
    if nav:
        for a in nav.find_all("a", class_="list_comments_link"):
            href = a.get("href", "")
            m = re.search(r"\?p=(\d+)", href)
            if m:
                chapters.append({
                    "pid": int(m.group(1)),
                    "title": a.get("title", "")
                })
    return chapters


def scrape_chapter(pid: int, base: str, session: requests.Session, max_pages: int = 300):
    rows = []
    prev_first = None

    for cpage in range(1, max_pages + 1):
        if st.session_state.abort:
            return rows

        r = session.get(f"{base}?p={pid}&cpage={cpage}#comments", timeout=25)
        soup = BeautifulSoup(r.text, "html5lib")
        comments = soup.select("ul.comment_list > li.comment")

        if not comments:
            break

        first_id = comments[0].get("id")
        if first_id == prev_first:
            break
        prev_first = first_id

        for li in comments:
            if st.session_state.abort:
                return rows

            cid = li.get("id", "").replace("comment-", "").strip()
            user_block = li.find("div", class_="user")
            if user_block:
                user_block.extract()
            text = li.get_text("\n", strip=True)

            rows.append({
                "chapter_p": pid,
                "comment_id": cid,
                "text": text
            })

        time.sleep(0.12)  # gentle throttling

    return rows


def scrape_consultation_with_progress(parent_id: str, base: str, translations: dict):
    """
    Full interactive scraping:
    - shows progress bar
    - abort checks
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; ConsultationAnalyzer/1.0)"
    })

    timing_info = scrape_consultation_timing(parent_id, base, session, translations)
    chapters = get_chapters(parent_id, base, session)
    all_rows = []

    prog = st.progress(0.0)
    status = st.empty()

    total = max(1, len(chapters))
    for i, ch in enumerate(chapters, start=1):
        if st.session_state.abort:
            break

        pid = ch["pid"]
        status.write(f"{translations.get('scraping_chapter','Scraping chapter')} {i}/{len(chapters)} (p={pid})")

        rows = scrape_chapter(pid, base, session)
        all_rows.extend(rows)

        prog.progress(i / total)

    prog.empty()
    status.empty()

    return pd.DataFrame(all_rows), chapters, timing_info

# =========================================================
# DUPLICATE DETECTION (FAST FUZZY + PROGRESS)
# =========================================================

def optimized_fuzzy_groups(df: pd.DataFrame, threshold: float, step_status=None):
    """
    Bucketed fuzzy grouping using Union-Find clustering.
    Guarantees transitive closure and monotonic threshold behavior.
    """

    texts = df["text_clean"].tolist()
    lengths = [len(t) for t in texts]
    buckets = df["bucket"].tolist()

    n = len(texts)

    # ---------------------------
    # Union-Find (Disjoint Set)
    # ---------------------------

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            if rank[root_x] == rank[root_y]:
                rank[root_x] += 1

    # Progress bar
    progress_bar = st.progress(0.0)
    comparisons_done = 0
    approx_total = n  # representative progress metric

    # Map bucket -> indices
    bucket_map = {}
    for idx, b in enumerate(buckets):
        bucket_map.setdefault(b, []).append(idx)

    # ---------------------------
    # Similarity comparisons
    # ---------------------------

    for bucket_value in bucket_map.keys():

        candidate_buckets = [bucket_value - 1, bucket_value, bucket_value + 1]

        candidate_indices = []
        for cb in candidate_buckets:
            candidate_indices.extend(bucket_map.get(cb, []))

        # deduplicate indices
        seen = set()
        candidate_indices = [
            x for x in candidate_indices
            if not (x in seen or seen.add(x))
        ]

        for i_idx in range(len(candidate_indices)):
            i = candidate_indices[i_idx]

            if st.session_state.abort:
                progress_bar.empty()
                return {}, {}

            for j_idx in range(i_idx + 1, len(candidate_indices)):
                j = candidate_indices[j_idx]

                # length filter ±20%
                if abs(lengths[i] - lengths[j]) / max(lengths[i], lengths[j]) > 0.2:
                    continue

                if fuzz.ratio(texts[i], texts[j]) >= threshold:
                    union(i, j)

            comparisons_done += 1
            progress_bar.progress(min(comparisons_done / approx_total, 1.0))

    progress_bar.empty()

    # ---------------------------
    # Build clusters
    # ---------------------------

    clusters = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    group_sizes = {}
    group_ids = {}

    df["dup_size"] = 1

    for root, members in clusters.items():
        if len(members) > 1:
            representative = members[0]
            size = len(members)

            group_sizes[representative] = size
            group_ids[representative] = (
                df.loc[members, "comment_id"].astype(str).tolist()
            )

            df.loc[members, "dup_size"] = size

    return group_sizes, group_ids
