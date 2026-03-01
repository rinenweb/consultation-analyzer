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
# SESSION STATE (ABORT)
# ---------------------------

if "abort" not in st.session_state:
    st.session_state.abort = False

# ---------------------------
# LANGUAGE SWITCH (FLAGS)
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
            min_value=80,
            max_value=100,
            value=90
        )

run_button = st.button(T["run"])

# ---------------------------
# SCRAPER (CACHED 10 MIN)
# ---------------------------

def get_chapters(parent_id):
    base = "https://www.opengov.gr/minenv/"
    url = f"{base}?p={parent_id}"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html5lib")

    nav_ul = soup.find("ul", class_="other_posts")
    chapters = []

    if nav_ul:
        for a in nav_ul.find_all("a", class_="list_comments_link", href=True):
            match = re.search(r"\?p=(\d+)", a["href"])
            if match:
                pid = int(match.group(1))
                title = a.get("title", "")
                chapters.append({"pid": pid, "title": title})

    return chapters


@st.cache_data(ttl=600)
def run_scraping_cached(parent_id):

    base = "https://www.opengov.gr/minenv/"
    all_rows = []
    chapters = get_chapters(parent_id)
    max_pages = 300

    for ch in chapters:
        pid = ch["pid"]
        prev_first = None

        for cpage in range(1, max_pages):
            url = f"{base}?p={pid}&cpage={cpage}#comments"
            r = requests.get(url)
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
                all_rows.append({
                    "chapter_p": pid,
                    "comment_id": cid,
                    "text": text
                })

            time.sleep(0.2)

    return pd.DataFrame(all_rows), chapters


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ---------------------------
# RUN
# ---------------------------

if run_button and url_input:

    st.session_state.abort = False

    if "opengov.gr" not in url_input and not url_input.isdigit():
        st.error("Only opengov.gr consultations or valid parent IDs are allowed.")
        st.stop()

    parent_id = re.search(r"\?p=(\d+)", url_input).group(1) if "opengov.gr" in url_input else url_input

    col_spin, col_abort = st.columns([8,1])

    with col_abort:
        abort_clicked = st.button("Abort" if lang=="en" else "Ακύρωση")

    if abort_clicked:
        st.session_state.abort = True
        st.warning("Aborted." if lang=="en" else "Η διαδικασία ακυρώθηκε.")
        st.stop()

    with col_spin:
        with st.spinner(T["scraping"]):
            df, chapters = run_scraping_cached(parent_id)

    st.success(T["completed"])

    df["text_clean"] = df["text"].str.strip().str.lower()
    df["word_count"] = df["text"].str.split().apply(len)

    # ---------------------------
    # DUPLICATE DETECTION
    # ---------------------------

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

    campaign_share = round((df["dup_size"] > 1).mean()*100, 2)
    duplicate_templates = len(template_groups)

    # ---------------------------
    # STRICT LAYER
    # ---------------------------

    policy_patterns = [w.strip() for w in policy_keywords_input.split(",")]
    amend_patterns = [w.strip() for w in amendment_keywords_input.split(",")]

    df["mentions_article"] = df["text_clean"].str.contains("|".join(policy_patterns), regex=True)
    df["mentions_amendment"] = df["text_clean"].str.contains("|".join(amend_patterns), regex=True)

    strict_layer = round((df["mentions_article"] & df["mentions_amendment"]).mean() * 100, 2)

    # ---------------------------
    # CORE METRICS
    # ---------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(T["total"], len(df))

    col2.metric(T["campaign"], campaign_share, help=T["campaign_help"])

    col3.metric(T["templates"], duplicate_templates, help=T["templates_help"])

    col4.metric(T["strict"], strict_layer, help=T["strict_desc"])

    # ---------------------------
    # TEXT STATISTICS
    # ---------------------------

    st.subheader(T["stats"])

    mean = df["word_count"].mean()
    median = df["word_count"].median()
    std = df["word_count"].std()

    c1, c2, c3 = st.columns(3)
    c1.metric(T["mean"], round(mean,2), help=T["mean_help"])
    c2.metric(T["median"], round(median,2), help=T["median_help"])
    c3.metric(T["std"], round(std,2), help=T["std_help"])

    # ---------------------------
    # KDE PLOT
    # ---------------------------

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

    # ---------------------------
    # METHODOLOGICAL PANEL
    # ---------------------------

    with st.expander(T["method_panel"], expanded=False):

        st.markdown("### " + T["execution_summary"])

        chapter_counts = df.groupby("chapter_p").size().reset_index(name="Comment Count")
        chapter_df = pd.merge(
            pd.DataFrame(chapters),
            chapter_counts,
            left_on="pid",
            right_on="chapter_p",
            how="left"
        ).fillna(0)

        chapter_df = chapter_df[["pid", "title", "Comment Count"]]
        chapter_df.columns = T["chapter_table_headers"]

        st.dataframe(chapter_df, use_container_width=True)

        st.markdown("### " + T["active_configuration"])
        st.write("Duplicate detection method:", duplicate_method)

        if duplicate_method == T["fuzzy_match"]:
            st.write("Similarity threshold (%):", similarity_threshold)

        st.write("Stopwords:", stopwords_input)
        st.write("Policy keywords:", policy_keywords_input)
        st.write("Amendment verbs:", amendment_keywords_input)
        st.write(T["timestamp"] + ":", str(pd.Timestamp.now()))
