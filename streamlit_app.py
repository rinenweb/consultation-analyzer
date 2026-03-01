import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from collections import Counter
import time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from translations import TRANSLATIONS

st.set_page_config(page_title="Consultation Analyzer", layout="wide")

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
run_button = st.button(T["run"])

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

# ---------------------------
# SCRAPER
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


def run_scraping(parent_id):

    base = "https://www.opengov.gr/minenv/"
    all_rows = []
    chapters = get_chapters(parent_id)
    max_pages = 300

    progress = st.progress(0)
    status = st.empty()

    for i, ch in enumerate(chapters):
        pid = ch["pid"]
        status.write(f"Scraping chapter {i+1}/{len(chapters)} (p={pid})")
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

        progress.progress((i+1)/len(chapters))

    progress.empty()
    status.empty()

    return pd.DataFrame(all_rows), chapters


# ---------------------------
# RUN
# ---------------------------

if run_button and url_input:

    if "opengov.gr" not in url_input and not url_input.isdigit():
        st.error("Only opengov.gr consultations or valid parent IDs are allowed.")
        st.stop()

    parent_id = re.search(r"\?p=(\d+)", url_input).group(1) if "opengov.gr" in url_input else url_input

    with st.spinner(T["scraping"]):
        df, chapters = run_scraping(parent_id)

    st.success(T["completed"])

    df["text_clean"] = df["text"].str.strip().str.lower()
    dup_counts = df["text_clean"].value_counts()
    df["dup_size"] = df["text_clean"].map(dup_counts)
    df["word_count"] = df["text"].str.split().apply(len)

    campaign_share = round((df["dup_size"] > 1).mean()*100,2)
    duplicate_templates = int((dup_counts > 1).sum())

    # Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        T["total"],
        len(df)
    )

    col2.metric(
        T["campaign"],
        campaign_share,
        help=T["campaign_help"]
    )

    col3.metric(
        T["templates"],
        duplicate_templates,
        help=T["templates_help"]
    )

    # Statistics
    st.subheader(T["stats"])

    mean = df["word_count"].mean()
    median = df["word_count"].median()
    std = df["word_count"].std()

    c1, c2, c3 = st.columns(3)
    c1.metric(T["mean"], round(mean,2), help=T["mean_help"])
    c2.metric(T["median"], round(median,2), help=T["median_help"])
    c3.metric(T["std"], round(std,2), help=T["std_help"])

    # KDE Plot
    st.subheader(T["distribution"])

    kde = gaussian_kde(df["word_count"])
    x = np.linspace(df["word_count"].min(), df["word_count"].max(), 500)
    y = kde(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Density")
    ax.axvline(mean, linestyle="--", label="Mean")
    ax.axvline(median, linestyle=":", label="Median")
    ax.legend()
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

        st.markdown("### " + T["metric_definitions"])
        st.markdown(T["definitions_text"])

        st.markdown("### " + T["active_configuration"])
        st.write("Stopwords:", stopwords_input)
        st.write("Policy keywords:", policy_keywords_input)
        st.write("Amendment verbs:", amendment_keywords_input)
        st.write(T["timestamp"] + ":", str(pd.Timestamp.now()))
