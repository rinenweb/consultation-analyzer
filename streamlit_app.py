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

if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ---------------------------
# LANGUAGE SWITCH
# ---------------------------

col_left, col_right = st.columns([10,1])
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

def get_chapters(parent_id):
    base = "https://www.opengov.gr/minenv/"
    r = requests.get(f"{base}?p={parent_id}")
    soup = BeautifulSoup(r.text, "html5lib")

    nav_ul = soup.find("ul", class_="other_posts")
    chapters = []

    if nav_ul:
        for a in nav_ul.find_all("a", class_="list_comments_link"):
            match = re.search(r"\?p=(\d+)", a["href"])
            if match:
                chapters.append({
                    "pid": int(match.group(1)),
                    "title": a.get("title", "")
                })

    return chapters


def run_scraping(parent_id):

    base = "https://www.opengov.gr/minenv/"
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

            r = requests.get(f"{base}?p={pid}&cpage={cpage}#comments")
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

        progress.progress((i+1)/len(chapters))

    progress.empty()
    status.empty()

    return pd.DataFrame(rows), chapters


@st.cache_data(ttl=600, show_spinner=False)
def run_scraping_cached(parent_id):
    return run_scraping(parent_id)


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ---------------------------
# RUN
# ---------------------------

if run_button and url_input:

    st.session_state.abort = False
    st.session_state.is_running = True

    if "opengov.gr" not in url_input and not url_input.isdigit():
        st.error("Only opengov.gr consultations or valid parent IDs are allowed.")
        st.stop()

    parent_id = re.search(r"\?p=(\d+)", url_input).group(1) if "opengov.gr" in url_input else url_input

    # Abort button only during scraping
    col_spin, col_abort = st.columns([8,1])

    with col_abort:
        if st.session_state.is_running:
            abort_clicked = st.button("Abort" if lang=="en" else "Ακύρωση")
            if abort_clicked:
                st.session_state.abort = True
                st.warning("Aborted." if lang=="en" else "Η διαδικασία ακυρώθηκε.")
                st.session_state.is_running = False
                st.stop()

    start_time = time.time()

    with col_spin:
        with st.spinner(T["scraping"]):
            df, chapters = run_scraping_cached(parent_id)

    duration = time.time() - start_time

    if duration < 0.5:
        st.info("Results loaded from cache." if lang=="en" else "Τα αποτελέσματα φορτώθηκαν από cache.")

    # ---------------------------
    # ANALYSIS PHASE
    # ---------------------------

    analysis_status = st.empty()
    analysis_progress = st.progress(0)

    # Step 1 - Normalize
    analysis_status.write("Normalizing text...")
    df["text_clean"] = df["text"].str.strip().str.lower()
    df["word_count"] = df["text"].str.split().apply(len)
    analysis_progress.progress(20)

    # Step 2 - Duplicate detection
    analysis_status.write("Detecting duplicates...")
    if duplicate_method == T["exact_match"]:

        dup_counts = df["text_clean"].value_counts()
        df["dup_size"] = df["text_clean"].map(dup_counts)
        template_groups = dup_counts[dup_counts > 1]

    else:

        texts = df["text_clean"].tolist()
        used = set()
        group_sizes = {}
        threshold = similarity_threshold / 100

        for i, t1 in enumerate(texts):
            if i in used:
                continue
            group = [i]
            for j in range(i+1, len(texts)):
                if j in used:
                    continue
                if similarity(t1, texts[j]) >= threshold:
                    group.append(j)
                    used.add(j)
            if len(group) > 1:
                group_sizes[i] = len(group)

        df["dup_size"] = 1
        for idx, size in group_sizes.items():
            df.loc[idx, "dup_size"] = size

        template_groups = pd.Series(group_sizes)

    analysis_progress.progress(40)

    campaign_share = round((df["dup_size"] > 1).mean()*100, 2)
    duplicate_templates = len(template_groups)

    # Step 3 - Strict layer
    analysis_status.write("Calculating strict layer...")
    policy_patterns = [w.strip() for w in policy_keywords_input.split(",")]
    amend_patterns = [w.strip() for w in amendment_keywords_input.split(",")]

    df["mentions_article"] = df["text_clean"].str.contains("|".join(policy_patterns), regex=True)
    df["mentions_amendment"] = df["text_clean"].str.contains("|".join(amend_patterns), regex=True)

    strict_layer = round((df["mentions_article"] & df["mentions_amendment"]).mean()*100, 2)

    analysis_progress.progress(60)

    # Step 4 - Statistics
    analysis_status.write("Computing statistics...")
    mean = df["word_count"].mean()
    median = df["word_count"].median()
    std = df["word_count"].std()

    analysis_progress.progress(80)

    # Step 5 - Finalizing
    analysis_status.write("Finalizing results...")
    analysis_progress.progress(100)

    # Hide analysis panel
    analysis_status.empty()
    analysis_progress.empty()

    st.session_state.is_running = False

    st.success(T["completed"])

    # ---------------------------
    # DISPLAY RESULTS
    # ---------------------------

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(T["total"], len(df))
    col2.metric(T["campaign"], campaign_share, help=T["campaign_help"])
    col3.metric(T["templates"], duplicate_templates, help=T["templates_help"])
    col4.metric(T["strict"], strict_layer, help=T["strict_desc"])

    # Top templates
    if duplicate_templates > 0:
        with st.expander(T["top_templates"], expanded=False):

            top_templates = template_groups.sort_values(ascending=False).head(5)

            for idx, count in top_templates.items():
                text = idx if duplicate_method == T["exact_match"] else df.loc[idx, "text"]
                st.markdown(f"**{T['occurrences']}:** {count}")
                preview = text[:400] + ("..." if len(text) > 400 else "")
                st.write(preview)
                if len(text) > 400:
                    with st.expander(T["show_full_text"]):
                        st.write(text)
                st.markdown("---")

    # Text stats
    st.subheader(T["stats"])
    c1, c2, c3 = st.columns(3)
    c1.metric(T["mean"], round(mean,2), help=T["mean_help"])
    c2.metric(T["median"], round(median,2), help=T["median_help"])
    c3.metric(T["std"], round(std,2), help=T["std_help"])

    # KDE
    st.subheader(T["distribution"])
    fig, ax = plt.subplots(figsize=(10,4))
    kde = gaussian_kde(df["word_count"])
    x = np.linspace(df["word_count"].min(), df["word_count"].max(), 500)
    y = kde(x)

    ax.plot(x, y, label=T["density"])
    ax.axvline(mean, linestyle="--", label=T["mean_line"])
    ax.axvline(median, linestyle=":", label=T["median_line"])
    ax.set_xlabel(T["word_count_label"])
    ax.set_ylabel(T["density"])
    ax.legend()
    ax.tick_params(axis='y', labelleft=False)
    plt.tight_layout()
    st.pyplot(fig)
