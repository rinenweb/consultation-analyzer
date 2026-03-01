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

st.set_page_config(page_title="Consultation Analyzer", layout="wide")

# ---------------------------
# SESSION STATE
# ---------------------------

if "abort" not in st.session_state:
    st.session_state.abort = False

# results persist across reruns
if "results" not in st.session_state:
    st.session_state.results = None  # dict with df, chapters, metrics, templates etc.

# ---------------------------
# LANGUAGE SWITCH (FLAGS)
# ---------------------------

col_left, col_right = st.columns([10, 1])
with col_right:
    lang_flag = st.radio("", ["🇬🇧", "🇬🇷"], horizontal=True, label_visibility="collapsed")

lang = "en" if lang_flag == "🇬🇧" else "el"
T = TRANSLATIONS[lang]

# ---------------------------
# HEADER
# ---------------------------

st.title(T["title"])
st.markdown(T["subtitle"])

# ---------------------------
# INPUT
# ---------------------------

url_input = st.text_input(T["input_label"])

# ---------------------------
# ADVANCED SETTINGS
# ---------------------------

with st.expander(T["advanced"]):

    stopwords_input = st.text_area(
        T["stopwords"],
        "και,να,το,η,της,την,των,σε,με,για,που,από,στο,στη,στον,οι,ο,τα,τι,ως,είναι,δεν,θα,ή,του,μια,ένα,τους,στην,όπως,στις,έχει,αλλά"
    )

    policy_keywords_input = st.text_area(
        T["policy"],
        "άρθρ,συνταγμ,οδηγ,ευρωπαϊκ,εε,ενωσιακ"
    )

    amendment_keywords_input = st.text_area(
        T["amend"],
        "να προστεθ,να διαγραφ,να αντικατασταθ,να τροποποιηθ"
    )

    # Default = Fuzzy
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

# ---------------------------
# SCRAPER
# ---------------------------

BASE = "https://www.opengov.gr/minenv/"

def get_chapters(parent_id: str):
    r = requests.get(f"{BASE}?p={parent_id}")
    soup = BeautifulSoup(r.text, "html5lib")

    nav_ul = soup.find("ul", class_="other_posts")
    chapters = []

    if nav_ul:
        for a in nav_ul.find_all("a", class_="list_comments_link"):
            href = a.get("href", "")
            match = re.search(r"\?p=(\d+)", href)
            if match:
                chapters.append({
                    "pid": int(match.group(1)),
                    "title": a.get("title", "")
                })
    return chapters


def run_scraping(parent_id: str):
    chapters = get_chapters(parent_id)
    rows = []
    max_pages = 300

    progress = st.progress(0)
    status = st.empty()

    for i, ch in enumerate(chapters):

        if st.session_state.abort:
            progress.empty()
            status.empty()
            return pd.DataFrame(), chapters

        pid = ch["pid"]
        status.write(f"Scraping chapter {i+1}/{len(chapters)} (p={pid})")

        prev_first = None
        for cpage in range(1, max_pages):

            if st.session_state.abort:
                progress.empty()
                status.empty()
                return pd.DataFrame(), chapters

            r = requests.get(f"{BASE}?p={pid}&cpage={cpage}#comments")
            soup = BeautifulSoup(r.text, "html5lib")
            comments = soup.select("ul.comment_list > li.comment")

            if not comments:
                break

            first_id = comments[0].get("id")
            if first_id == prev_first:
                break
            prev_first = first_id

            for li in comments:
                cid = li.get("id", "").replace("comment-", "")
                user_block = li.find("div", class_="user")
                if user_block:
                    user_block.extract()
                text = li.get_text("\n", strip=True)

                rows.append({
                    "chapter_p": pid,
                    "comment_id": cid,
                    "text": text
                })

            time.sleep(0.2)

        progress.progress((i + 1) / max(1, len(chapters)))

    progress.empty()
    status.empty()

    return pd.DataFrame(rows), chapters


@st.cache_data(ttl=600, show_spinner=False)
def run_scraping_cached(parent_id: str):
    # no Streamlit UI in cached function
    # NOTE: This will not show progress; progress is shown in run_scraping(), so cache applies only on next rerun.
    # In practice: first run -> uncached -> progress shows; within TTL -> returns cached fast.
    return run_scraping(parent_id)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def render_analysis_steps(container, steps, active_index, done_set):
    """
    Render a clean checklist with spinner for current step.
    done_set contains indices of completed steps.
    """
    with container:
        for idx, label in enumerate(steps):
            if idx in done_set:
                st.markdown(f"✅ {label}")
            elif idx == active_index:
                st.markdown(f"⏳ {label}")
            else:
                st.markdown(f"▫️ {label}")


def build_comment_link(comment_id: str) -> str:
    # stable permalink pattern
    return f"{BASE}?c={comment_id}"


# ---------------------------
# RUN
# ---------------------------

if run_button and url_input:

    # reset abort at run start
    st.session_state.abort = False

    if "opengov.gr" not in url_input and not url_input.isdigit():
        st.error("Only opengov.gr consultations or valid parent IDs are allowed.")
        st.stop()

    parent_id = re.search(r"\?p=(\d+)", url_input).group(1) if "opengov.gr" in url_input else url_input

    # --- Scraping row: spinner left, abort right (only during scraping)
    col_spin, col_abort = st.columns([8, 1])
    spinner_placeholder = col_spin.empty()

    abort_placeholder = col_abort.empty()
    abort_clicked = abort_placeholder.button("Abort" if lang == "en" else "Ακύρωση")

    if abort_clicked:
        st.session_state.abort = True
        st.warning("Aborted." if lang == "en" else "Η διαδικασία ακυρώθηκε.")
        abort_placeholder.empty()
        st.stop()

    start_time = time.time()

    with spinner_placeholder:
        with st.spinner(T["scraping"]):
            df, chapters = run_scraping_cached(parent_id)

    duration = time.time() - start_time

    # hide abort right after scraping ends (so it won't remain visible)
    abort_placeholder.empty()

    if st.session_state.abort:
        st.warning("Aborted." if lang == "en" else "Η διαδικασία ακυρώθηκε.")
        st.stop()

    if duration < 0.5:
        st.info("Results loaded from cache." if lang == "en" else "Τα αποτελέσματα φορτώθηκαν από cache.")

    # --- Analysis steps list (no second progress bar)
    steps = [
        "Normalizing text..." if lang == "en" else "Κανονικοποίηση κειμένου...",
        "Detecting duplicates..." if lang == "en" else "Εντοπισμός διπλότυπων...",
        "Calculating strict layer..." if lang == "en" else "Υπολογισμός Strict Layer...",
        "Computing statistics..." if lang == "en" else "Υπολογισμός στατιστικών...",
        "Preparing outputs..." if lang == "en" else "Προετοιμασία αποτελεσμάτων..."
    ]

    steps_box = st.container()
    done = set()
    active = 0
    render_analysis_steps(steps_box, steps, active, done)

    # Step 1: Normalize
    df["text_clean"] = df["text"].str.strip().str.lower()
    df["word_count"] = df["text"].str.split().apply(len)
    done.add(0); active = 1
    render_analysis_steps(steps_box, steps, active, done)

    # Step 2: Duplicate detection (exact or fuzzy)
    template_groups = None
    template_group_ids = {}  # representative -> list of comment_ids (for linking)
    if duplicate_method == T["exact_match"]:
        dup_counts = df["text_clean"].value_counts()
        df["dup_size"] = df["text_clean"].map(dup_counts)
        template_groups = dup_counts[dup_counts > 1]

        # Map template text -> comment ids
        if len(template_groups) > 0:
            for template_text in template_groups.index.tolist():
                ids = df.loc[df["text_clean"] == template_text, "comment_id"].astype(str).tolist()
                template_group_ids[template_text] = ids

    else:
        texts = df["text_clean"].tolist()
        used = set()
        group_sizes = {}
        threshold = similarity_threshold / 100

        # For fuzzy, representative is the index i of the first comment in the group
        for i, t1 in enumerate(texts):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(texts)):
                if j in used:
                    continue
                if similarity(t1, texts[j]) >= threshold:
                    group.append(j)
                    used.add(j)
            if len(group) > 1:
                group_sizes[i] = len(group)
                # collect comment ids for transparency
                template_group_ids[i] = df.loc[group, "comment_id"].astype(str).tolist()

        df["dup_size"] = 1
        for idx, size in group_sizes.items():
            df.loc[idx, "dup_size"] = size

        template_groups = pd.Series(group_sizes)

    campaign_share = round((df["dup_size"] > 1).mean() * 100, 2)
    duplicate_templates = len(template_groups)

    done.add(1); active = 2
    render_analysis_steps(steps_box, steps, active, done)

    # Step 3: Strict layer
    policy_patterns = [w.strip() for w in policy_keywords_input.split(",") if w.strip()]
    amend_patterns = [w.strip() for w in amendment_keywords_input.split(",") if w.strip()]

    df["mentions_article"] = df["text_clean"].str.contains("|".join(policy_patterns), regex=True) if policy_patterns else False
    df["mentions_amendment"] = df["text_clean"].str.contains("|".join(amend_patterns), regex=True) if amend_patterns else False

    strict_layer = round((df["mentions_article"] & df["mentions_amendment"]).mean() * 100, 2)

    done.add(2); active = 3
    render_analysis_steps(steps_box, steps, active, done)

    # Step 4: Statistics
    mean = df["word_count"].mean()
    median = df["word_count"].median()
    std = df["word_count"].std()

    done.add(3); active = 4
    render_analysis_steps(steps_box, steps, active, done)

    # Step 5: Prepare outputs (KDE etc.)
    # (we compute KDE later at display time to keep analysis stage minimal)
    done.add(4)
    render_analysis_steps(steps_box, steps, active, done)

    # Hide analysis panel
    steps_box.empty()

    # Persist results in session state so they remain after reruns
    st.session_state.results = {
        "df": df,
        "chapters": chapters,
        "campaign_share": campaign_share,
        "duplicate_templates": duplicate_templates,
        "strict_layer": strict_layer,
        "mean": mean,
        "median": median,
        "std": std,
        "template_groups": template_groups,
        "template_group_ids": template_group_ids,
        "duplicate_method": duplicate_method,
        "similarity_threshold": similarity_threshold,
        "stopwords_input": stopwords_input,
        "policy_keywords_input": policy_keywords_input,
        "amendment_keywords_input": amendment_keywords_input,
        "timestamp": str(pd.Timestamp.now())
    }

    st.success(T["completed"])

# ---------------------------
# DISPLAY (PERSISTENT RESULTS)
# ---------------------------

if st.session_state.results is not None:
    R = st.session_state.results
    df = R["df"]

    # Core metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(T["total"], len(df))
    col2.metric(T["campaign"], R["campaign_share"], help=T["campaign_help"])
    col3.metric(T["templates"], R["duplicate_templates"], help=T["templates_help"])
    col4.metric(T["strict"], R["strict_layer"], help=T["strict_desc"])

    # Top templates
    if R["duplicate_templates"] > 0:
        with st.expander(T["top_templates"], expanded=False):

            top_templates = R["template_groups"].sort_values(ascending=False).head(5)

            for key, count in top_templates.items():

                if R["duplicate_method"] == T["exact_match"]:
                    # key is the template text
                    full_text = key
                    ids = R["template_group_ids"].get(key, [])
                else:
                    # key is representative row index
                    full_text = df.loc[int(key), "text"]
                    ids = R["template_group_ids"].get(key, [])

                st.markdown(f"**{T['occurrences']}:** {int(count)}")

                preview = full_text[:400] + ("..." if len(full_text) > 400 else "")
                st.write(preview)

                with st.expander(T["show_full_text"]):
                    st.write(full_text)

                    # Links for transparency (up to 10)
                    ids = [str(x) for x in ids if str(x).strip()]
                    if ids:
                        to_show = ids[:10]
                        more = len(ids) - len(to_show)

                        links = []
                        for cid in to_show:
                            url = build_comment_link(cid)
                            links.append(f"[{cid}]({url})")
                        st.markdown(" ".join(links))

                        if more > 0:
                            st.caption(f"+{more} more")

                st.markdown("---")

    # Text stats
    st.subheader(T["stats"])
    c1, c2, c3 = st.columns(3)
    c1.metric(T["mean"], round(R["mean"], 2), help=T["mean_help"])
    c2.metric(T["median"], round(R["median"], 2), help=T["median_help"])
    c3.metric(T["std"], round(R["std"], 2), help=T["std_help"])

    # KDE plot
    st.subheader(T["distribution"])
    fig, ax = plt.subplots(figsize=(10, 4))

    kde = gaussian_kde(df["word_count"])
    x = np.linspace(df["word_count"].min(), df["word_count"].max(), 500)
    y = kde(x)

    ax.plot(x, y, label=T["density"])
    ax.axvline(R["mean"], linestyle="--", label=T["mean_line"])
    ax.axvline(R["median"], linestyle=":", label=T["median_line"])
    ax.set_xlabel(T["word_count_label"])
    ax.set_ylabel(T["density"])
    ax.legend()
    ax.tick_params(axis="y", labelleft=False)
    plt.tight_layout()
    st.pyplot(fig)

    # Methodological panel (kept)
    with st.expander(T["method_panel"], expanded=False):
        st.markdown("### " + T["execution_summary"])

        chapter_counts = df.groupby("chapter_p").size().reset_index(name="Comment Count")
        chapter_df = pd.merge(
            pd.DataFrame(R["chapters"]),
            chapter_counts,
            left_on="pid",
            right_on="chapter_p",
            how="left"
        ).fillna(0)

        chapter_df = chapter_df[["pid", "title", "Comment Count"]]
        chapter_df.columns = T["chapter_table_headers"]

        st.dataframe(chapter_df, use_container_width=True)

        st.markdown("### " + T["active_configuration"])
        st.write("Duplicate detection method:", R["duplicate_method"])
        if R["duplicate_method"] == T["fuzzy_match"]:
            st.write("Similarity threshold (%):", R["similarity_threshold"])
        st.write("Stopwords:", R["stopwords_input"])
        st.write("Policy keywords:", R["policy_keywords_input"])
        st.write("Amendment verbs:", R["amendment_keywords_input"])
        st.write(T["timestamp"] + ":", R["timestamp"])
