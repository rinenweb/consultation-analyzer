import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from difflib import SequenceMatcher
from translations import TRANSLATIONS
import json

st.set_page_config(page_title="Consultation Analyzer", layout="wide")

# =========================================================
# SESSION STATE
# =========================================================

if "abort" not in st.session_state:
    st.session_state.abort = False

if "results" not in st.session_state:
    st.session_state.results = None

if "running" not in st.session_state:
    st.session_state.running = False

# =========================================================
# LANGUAGE (FLAGS)
# =========================================================

col_left, col_right = st.columns([10, 1])
with col_right:
    lang_flag = st.radio("", ["🇬🇧", "🇬🇷"], horizontal=True, label_visibility="collapsed")

lang = "en" if lang_flag == "🇬🇧" else "el"
T = TRANSLATIONS[lang]

# =========================================================
# HEADER
# =========================================================

st.title(T["title"])
st.markdown(T["subtitle"])

# =========================================================
# INPUT + ADVANCED
# =========================================================

url_input = st.text_input(T["input_label"])

with st.expander(T["advanced"]):
    policy_keywords_input = st.text_area(
        T["policy"],
        "άρθρ,συνταγμ,οδηγ,ευρωπαϊκ,εε,ενωσιακ"
    )
    amendment_keywords_input = st.text_area(
        T["amend"],
        "να προστεθ,να διαγραφ,να αντικατασταθ,να τροποποιηθ"
    )

    duplicate_method = st.radio(
        T["duplicate_method_label"],
        [T["fuzzy_match"], T["exact_match"]],
        horizontal=True,
        index=0
    )

    similarity_threshold = 90
    if duplicate_method == T["fuzzy_match"]:
        similarity_threshold = st.slider(
            T["similarity_threshold"],
            80, 100, 90
        )

run_button = st.button(T["run"])

# =========================================================
# UTIL: EXTRACT BASE + PARENT ID
# =========================================================

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
    """
    Produces a canonical version of the comment text
    for duplicate detection purposes.

    Removes:
    - trailing numeric ID blocks
    - '+X more' tails
    - excessive whitespace
    """

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # remove trailing lines containing only numbers (IDs)
    text = re.sub(r"\n?\s*(\d+\s+)+\d+\s*", "", text)

    # remove '+X more' fragments
    text = re.sub(r"\+?\d+\s*more", "", text)

    # normalize whitespace
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


def scrape_consultation_with_progress(parent_id: str, base: str):
    """
    Full interactive scraping:
    - shows progress bar
    - abort checks
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; ConsultationAnalyzer/1.0)"
    })

    chapters = get_chapters(parent_id, base, session)
    all_rows = []

    prog = st.progress(0.0)
    status = st.empty()

    total = max(1, len(chapters))
    for i, ch in enumerate(chapters, start=1):
        if st.session_state.abort:
            break

        pid = ch["pid"]
        status.write(f"{T.get('scraping_chapter','Scraping chapter')} {i}/{len(chapters)} (p={pid})")

        rows = scrape_chapter(pid, base, session)
        all_rows.extend(rows)

        prog.progress(i / total)

    prog.empty()
    status.empty()

    return pd.DataFrame(all_rows), chapters

# =========================================================
# DUPLICATE DETECTION (FAST FUZZY + OPTIONAL PROGRESS)
# =========================================================

def optimized_fuzzy_groups(df: pd.DataFrame, threshold: float, step_status=None):
    """
    Returns:
      group_sizes: dict[rep_index -> size]
      group_ids: dict[rep_index -> list(comment_ids)]
    Strategy:
      - bucket by first 40 chars
      - length ratio filter (<=20%)
      - SequenceMatcher only within bucket
    """
    texts = df["text_clean"].tolist()
    buckets = {}

    for idx, text in enumerate(texts):
        key = text[:40]
        buckets.setdefault(key, []).append(idx)

    group_sizes = {}
    group_ids = {}

    # progress over buckets (not too noisy)
    bucket_keys = list(buckets.keys())
    n_buckets = max(1, len(bucket_keys))

    for bi, key in enumerate(bucket_keys, start=1):
        if st.session_state.abort:
            break

        if step_status is not None and (bi % 10 == 0 or bi == 1 or bi == n_buckets):
            step_status.write(
                f"{T.get('dup_progress','Detecting duplicates')}… {bi}/{n_buckets}"
            )

        bucket = buckets[key]
        # compare within bucket
        for i in range(len(bucket)):
            if st.session_state.abort:
                break

            idx_i = bucket[i]
            if idx_i in group_sizes:
                continue

            t1 = texts[idx_i]
            group = [idx_i]

            for j in bucket[i+1:]:
                if st.session_state.abort:
                    break

                t2 = texts[j]

                # length ratio filter
                if abs(len(t1) - len(t2)) / max(len(t1), len(t2)) > 0.2:
                    continue

                if SequenceMatcher(None, t1, t2).ratio() >= threshold:
                    group.append(j)

            if len(group) > 1:
                group_sizes[idx_i] = len(group)
                group_ids[idx_i] = df.loc[group, "comment_id"].astype(str).tolist()

    df["dup_size"] = 1
    for rep_idx, size in group_sizes.items():
        df.loc[rep_idx, "dup_size"] = size

    return group_sizes, group_ids

# =========================================================
# RUN
# =========================================================

if run_button and url_input:

    st.session_state.abort = False
    st.session_state.running = True
    st.session_state.results = None

    base, parent_id = extract_base_and_parent(url_input)
    if not base:
        st.error(T.get("invalid_url", "Please provide a valid full opengov consultation URL."))
        st.session_state.running = False
        st.stop()

    # --- Abort button (visible ONLY while running)
    col_run, col_abort = st.columns([10, 1])
    with col_abort:
        if st.button(T.get("abort", "Abort")):
            st.session_state.abort = True
            st.session_state.running = False
            st.warning(T.get("aborted_msg", "Aborted."))
            st.stop()

    # --- Scraping message (will be cleared later)
    scrape_msg = st.empty()
    scrape_msg.info(T["scraping"])

    # --- Step panel
    steps = [
        T.get("step_scrape", "Scraping chapters & comments"),
        T.get("step_normalize", "Normalizing text"),
        T.get("step_duplicates", "Detecting duplicates"),
        T.get("step_strict", "Calculating strict layer"),
        T.get("step_stats", "Computing statistics"),
        T.get("step_render", "Preparing outputs"),
    ]

    steps_ph = st.empty()
    done = set()
    active = 0
    render_steps(steps_ph, steps, active, done)

    # ================= STEP 1: SCRAPE =================

    df, chapters = scrape_consultation_with_progress(parent_id, base)

    if st.session_state.abort:
        scrape_msg.empty()
        steps_ph.empty()
        st.session_state.running = False
        st.stop()

    if df.empty:
        scrape_msg.empty()
        steps_ph.empty()
        st.session_state.running = False
        st.warning(T.get("no_comments", "No comments found."))
        st.stop()

    done.add(0)
    active = 1
    render_steps(steps_ph, steps, active, done)

    # ================= STEP 2: NORMALIZE =================

    # canonical text for duplicate detection
    df["text_clean"] = df["text"].astype(str).apply(canonicalize_text)

    # word count based on original text (not canonicalized)
    df["word_count"] = df["text"].astype(str).str.split().apply(len)

    done.add(1)
    active = 2
    render_steps(steps_ph, steps, active, done)

    # ================= STEP 3: DUPLICATES =================

    dup_status = st.empty()

    template_ids = {}

    if duplicate_method == T["exact_match"]:
        dup_counts = df["text_clean"].value_counts()
        df["dup_size"] = df["text_clean"].map(dup_counts)
        template_groups = dup_counts[dup_counts > 1]

        for template_text in template_groups.index.tolist():
            ids = df.loc[df["text_clean"] == template_text, "comment_id"].astype(str).tolist()
            template_ids[template_text] = ids

    else:
        group_sizes, group_ids = optimized_fuzzy_groups(
            df,
            similarity_threshold / 100.0,
            step_status=dup_status
        )
        template_groups = pd.Series(group_sizes).sort_values(ascending=False)
        template_ids = group_ids

    dup_status.empty()

    if st.session_state.abort:
        scrape_msg.empty()
        steps_ph.empty()
        st.session_state.running = False
        st.stop()

    campaign_share = round((df["dup_size"] > 1).mean() * 100, 2)
    duplicate_templates = int(len(template_groups))

    done.add(2)
    active = 3
    render_steps(steps_ph, steps, active, done)

    # ================= STEP 4: STRICT =================

    policy_patterns = [w.strip() for w in policy_keywords_input.split(",") if w.strip()]
    amend_patterns = [w.strip() for w in amendment_keywords_input.split(",") if w.strip()]

    if policy_patterns:
        df["mentions_article"] = df["text_clean"].str.contains("|".join(policy_patterns), regex=True)
    else:
        df["mentions_article"] = False

    if amend_patterns:
        df["mentions_amendment"] = df["text_clean"].str.contains("|".join(amend_patterns), regex=True)
    else:
        df["mentions_amendment"] = False

    strict_layer = round((df["mentions_article"] & df["mentions_amendment"]).mean() * 100, 2)

    done.add(3)
    active = 4
    render_steps(steps_ph, steps, active, done)

    # ================= STEP 5: STATS =================

    mean = float(df["word_count"].mean())
    median = float(df["word_count"].median())
    max_words = int(df["word_count"].max())
    std = float(df["word_count"].std(ddof=1)) if len(df) > 1 else 0.0

    done.add(4)
    active = 5
    render_steps(steps_ph, steps, active, done)

    # ================= STORE RESULTS =================

    st.session_state.results = {
        "df": df,
        "chapters": chapters,
        "base": base,
        "parent_id": parent_id,
        "duplicate_method": duplicate_method,
        "similarity_threshold": similarity_threshold,
        "policy_keywords": policy_keywords_input,
        "amendment_keywords": amendment_keywords_input,
        "campaign_share": campaign_share,
        "duplicate_templates": duplicate_templates,
        "strict_layer": strict_layer,
        "mean": mean,
        "median": median,
        "max": max_words,
        "std": std,
        "template_groups": template_groups,
        "template_ids": template_ids,
        "timestamp": str(pd.Timestamp.now())
    }

    done.add(5)
    render_steps(steps_ph, steps, active, done)

    # ================= CLEAN UI =================

    time.sleep(0.15)
    scrape_msg.empty()
    steps_ph.empty()

    st.session_state.running = False

    st.success(T.get("completed", "Analysis completed."))

# =========================================================
# DISPLAY RESULTS (PERSIST)
# =========================================================

if st.session_state.results:
    R = st.session_state.results
    df = R["df"]
    base = R["base"]

    # --- CORE METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric(T.get("total", "Total Comments"), len(df))
    c2.metric(
        T["campaign"],
        R["campaign_share"],
        help=T.get("campaign_help", None)
    )
    c3.metric(
        T["duplicate_templates"],
        R["duplicate_templates"],
        help=T["duplicate_templates_help"]
    )
    c4.metric(
        T.get("strict", "Strict Layer"),
        R["strict_layer"],
        help=T.get("strict_desc", None)
    )

    # --- TEXT STATISTICS ---
    st.subheader(T.get("stats", "Text Statistics"))
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(T.get("mean", "Mean words"), round(R["mean"], 2), help=T.get("mean_help", None))
    s2.metric(T.get("median", "Median words"), round(R["median"], 2), help=T.get("median_help", None))
    s3.metric(T.get("max", "Max words"), R["max"])
    s4.metric(T.get("std", "Std deviation"), round(R["std"], 2), help=T.get("std_help", None))

    # --- TOP TEMPLATES ---
    if R["duplicate_templates"] > 0:
        with st.expander(T.get("top_templates", "Top Duplicate Templates"), expanded=False):
            # show up to 5
            if isinstance(R["template_groups"], pd.Series):
                top = R["template_groups"].sort_values(ascending=False).head(5)
                items = list(top.items())
            else:
                # exact match returns Series too, but just in case
                items = list(R["template_groups"].items())[:5]

            for key, count in items:
                count_int = int(count)

                # exact: key is template text; fuzzy: key is representative index
                if R["duplicate_method"] == T["exact_match"]:
                    full_text = str(key)
                    ids = R["template_ids"].get(key, [])
                else:
                    rep_idx = int(key)
                    full_text = df.loc[rep_idx, "text"]
                    ids = R["template_ids"].get(rep_idx, [])

                st.markdown(f"**{T.get('occurrences','Occurrences')}: {count_int}**")

                preview = full_text[:400] + ("..." if len(full_text) > 400 else "")
                st.write(preview)

                with st.expander(T.get("show_full_text", "Show full text")):
                    st.write(full_text)

                    ids = [str(x) for x in ids if str(x).strip()]
                    to_show = ids[:10]
                    more = len(ids) - len(to_show)

                    if to_show:
                        links = [f"[{cid}]({build_comment_link(base, cid)})" for cid in to_show]
                        st.markdown(" ".join(links))
                        if more > 0:
                            st.caption(f"+{more} more")

                st.markdown("---")

    # --- KDE PLOT ---
    if len(df) > 1:
        st.subheader(T.get("distribution", "Comment Length Distribution (KDE)"))

        fig, ax = plt.subplots(figsize=(10, 4))
        kde = gaussian_kde(df["word_count"])
        x = np.linspace(df["word_count"].min(), df["word_count"].max(), 500)
        y = kde(x)

        ax.plot(x, y, label=T.get("density", "Density"))
        ax.axvline(R["mean"], linestyle="--", label=T.get("mean_line", "Mean"))
        ax.axvline(R["median"], linestyle=":", label=T.get("median_line", "Median"))
        ax.set_xlabel(T.get("word_count_label", "Word count"))
        ax.legend()
        st.pyplot(fig)
    
    # ---------------------------
    # EXPORTS (CSV + METADATA)
    # ---------------------------
    
    exp1, exp2 = st.columns([1, 1])
    
    # 1) Comments CSV export (enriched)
    export_df = df.copy()
    
    # ensure columns exist (safety)
    for col in ["dup_size", "mentions_article", "mentions_amendment"]:
        if col not in export_df.columns:
            export_df[col] = False
    
    export_df["is_duplicate"] = export_df.get("dup_size", 1) > 1
    export_df["strict_flag"] = export_df.get("mentions_article", False) & export_df.get("mentions_amendment", False)
    
    # stable column order
    cols_order = [
        "chapter_p", "comment_id", "text",
        "word_count",
        "dup_size", "is_duplicate",
        "mentions_article", "mentions_amendment", "strict_flag"
    ]
    cols_order = [c for c in cols_order if c in export_df.columns]
    export_df = export_df[cols_order]
    
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    
    with exp1:
        st.download_button(
            label=T.get("export_comments_csv", "Export comments (CSV)"),
            data=csv_bytes,
            file_name=f"comments_{R.get('parent_id','consultation')}.csv",
            mime="text/csv"
        )
    
    # 2) Metadata JSON export (reproducibility)
    metadata = {
        "base": R.get("base"),
        "parent_id": R.get("parent_id"),
        "timestamp": R.get("timestamp"),
        "total_comments": int(len(df)),
        "total_chapters": int(len(R.get("chapters", []))),
        "duplicate_method": R.get("duplicate_method"),
        "similarity_threshold": R.get("similarity_threshold"),
        "campaign_share_pct": float(R.get("campaign_share")),
        "duplicate_templates": int(R.get("duplicate_templates")),
        "strict_layer_pct": float(R.get("strict_layer")),
        "policy_keywords": R.get("policy_keywords"),
        "amendment_keywords": R.get("amendment_keywords"),
        "text_stats": {
            "mean_words": float(R.get("mean")),
            "median_words": float(R.get("median")),
            "max_words": int(R.get("max")),
            "std_words": float(R.get("std")),
        },
        "tool": {
            "name": "consultation-analyzer",
            "repo": "https://github.com/rinenweb/consultation-analyzer/",
            "version": "wip"
        }
    }
    
    meta_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
    
    with exp2:
        st.download_button(
            label=T.get("export_metadata_json", "Export analysis metadata (JSON)"),
            data=meta_bytes,
            file_name=f"metadata_{R.get('parent_id','consultation')}.json",
            mime="application/json"
        )
    
    # --- METHODOLOGICAL PANEL ---
    with st.expander(T.get("method_panel", "Methodological Parameters & Execution Transparency"), expanded=False):
        st.markdown(f"### {T.get('execution_summary','Execution Summary')}")

        # chapters table with comment counts
        ch_df = pd.DataFrame(R["chapters"])
        if not ch_df.empty:
            counts = df.groupby("chapter_p").size().reset_index(name="Comment Count")
            merged = ch_df.merge(counts, left_on="pid", right_on="chapter_p", how="left").fillna(0)
            merged = merged[["pid", "title", "Comment Count"]]
            merged.columns = T.get("chapter_table_headers", ["Chapter ID", "Chapter Title", "Comment Count"])
            st.dataframe(merged, use_container_width=True)

        st.markdown(f"### {T.get('active_configuration','Active Configuration')}")
        st.write("BASE:", R["base"])
        st.write("Parent ID:", R["parent_id"])
        st.write(T.get("duplicate_method_label","Duplicate detection method") + ":", R["duplicate_method"])
        if R["duplicate_method"] == T["fuzzy_match"]:
            st.write(T.get("similarity_threshold","Similarity threshold (%)") + ":", R["similarity_threshold"])
        st.write(T.get("policy","Policy keywords") + ":", R["policy_keywords"])
        st.write(T.get("amend","Amendment verbs") + ":", R["amendment_keywords"])
        st.write(T.get("timestamp","Run timestamp") + ":", R["timestamp"])

# =========================================================
# FOOTER (Centered, Bilingual)
# =========================================================

st.markdown("---")

st.markdown(
    f"""
    <div style="text-align: center; font-size: 0.85rem; opacity: 0.8; max-width: 800px; margin: auto;">
        <p>{T["disclaimer_text"]}</p>
        <p style="margin-top: 6px;">{T["methodology_note"]}</p>
        <p style="margin-top: 6px;">{T["code_available"]}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style="text-align: center; margin-top: 12px;">
        {T["developed_by"]} 
        <a href="https://www.rinenweb.eu/" target="_blank" title="Rinenweb Development">
            <img style="padding-bottom: 5px; margin-left: 6px;"
                 src="https://www.rinenweb.eu/images/rinenweb-logo-color-sm.png"
                 alt="Rinenweb Logo"
                 width="100"
                 height="25">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)




