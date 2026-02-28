import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from collections import Counter
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Consultation Analyzer", layout="wide")

st.title("Public Consultation Analyzer")
st.markdown("Rule-based transparency tool for consultation analysis.")

# ---------------------------
# ADVANCED SETTINGS
# ---------------------------

with st.expander("Advanced Settings"):

    stopwords_input = st.text_area(
        "Stopwords (comma separated)",
        "και,να,το,η,της,την,των,σε,με,για,που,από,στο,στη,στον,οι,ο,τα,τι,ως,είναι,δεν,θα,ή,του,μια,ένα,τους,στην,όπως"
    )

    policy_keywords_input = st.text_area(
        "Policy keywords (comma separated)",
        "άρθρ,συνταγμ,οδηγ,ευρωπαϊκ,εε,ενωσιακ"
    )

    amendment_keywords_input = st.text_area(
        "Amendment verbs (comma separated)",
        "να προστεθ,να διαγραφ,να αντικατασταθ,να τροποποιηθ"
    )

# ---------------------------
# INPUT
# ---------------------------

url_input = st.text_input("Enter consultation URL or parent ID")
run_button = st.button("Run Analysis")

# ---------------------------
# SCRAPER
# ---------------------------

def get_chapter_pids(parent_id):
    base = "https://www.opengov.gr/minenv/"
    url = f"{base}?p={parent_id}"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html5lib")

    nav_ul = soup.find("ul", class_="other_posts")
    pids = []

    if nav_ul:
        for a in nav_ul.find_all("a", class_="list_comments_link", href=True):
            match = re.search(r"\?p=(\d+)", a["href"])
            if match:
                pids.append(int(match.group(1)))

    return sorted(pids)


def run_scraping(parent_id):

    base = "https://www.opengov.gr/minenv/"
    all_rows = []

    chapter_pids = get_chapter_pids(parent_id)
    max_pages = 300

    progress = st.progress(0)
    status = st.empty()

    total_steps = len(chapter_pids)
    current_step = 0

    for pid in chapter_pids:
        current_step += 1
        status.write(f"Scraping chapter {current_step}/{total_steps} (p={pid})")

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

        progress.progress(current_step / total_steps)

    progress.empty()
    status.empty()

    return pd.DataFrame(all_rows), chapter_pids


# ---------------------------
# ANALYSIS
# ---------------------------

def analyze(df):

    df["text_clean"] = df["text"].str.strip().str.lower()

    dup_counts = df["text_clean"].value_counts()
    df["dup_size"] = df["text_clean"].map(dup_counts)

    campaign_share = round((df["dup_size"] > 1).mean() * 100, 2)
    duplicate_templates = int((dup_counts > 1).sum())

    df["word_count"] = df["text"].str.split().apply(len)

    mean_words = round(df["word_count"].mean(), 2)
    median_words = int(df["word_count"].median())
    max_words = int(df["word_count"].max())
    std_words = round(df["word_count"].std(), 2)

    stopwords = set([w.strip() for w in stopwords_input.split(",")])
    words = []

    for text in df["text_clean"]:
        clean = re.sub(r"[^α-ωάέήίόύώϊϋΐΰa-z\s]", " ", text)
        for w in clean.split():
            if w not in stopwords and len(w) > 3:
                words.append(w)

    top_words = Counter(words).most_common(15)

    policy_patterns = [w.strip() for w in policy_keywords_input.split(",")]
    amend_patterns = [w.strip() for w in amendment_keywords_input.split(",")]

    df["mentions_article"] = df["text_clean"].str.contains("|".join(policy_patterns), regex=True)
    df["mentions_amendment"] = df["text_clean"].str.contains("|".join(amend_patterns), regex=True)

    strict_layer = round((df["mentions_article"] & df["mentions_amendment"]).mean() * 100, 2)

    return df, {
        "total_comments": len(df),
        "campaign_share": campaign_share,
        "duplicate_templates": duplicate_templates,
        "mean_words": mean_words,
        "median_words": median_words,
        "max_words": max_words,
        "std_words": std_words,
        "top_words": top_words,
        "strict_layer": strict_layer
    }


# ---------------------------
# RUN
# ---------------------------

if run_button and url_input:

    if "opengov.gr" not in url_input and not url_input.isdigit():
        st.error("Only opengov.gr consultations or valid parent IDs are allowed.")
        st.stop()

    if "opengov.gr" in url_input:
        match = re.search(r"\?p=(\d+)", url_input)
        if match:
            parent_id = match.group(1)
        else:
            st.error("Could not detect parent ID.")
            st.stop()
    else:
        parent_id = url_input

    with st.spinner("Scraping and analyzing consultation..."):
        df, chapters = run_scraping(parent_id)

        if df.empty:
            st.error("No comments detected.")
            st.stop()

        df, results = analyze(df)

    st.success("Analysis completed.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", results["total_comments"])
    col2.metric("Campaign Share (%)", results["campaign_share"])
    col3.metric("Duplicate Templates", results["duplicate_templates"])

    st.subheader("Text Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean words", results["mean_words"])
    c2.metric("Median words", results["median_words"])
    c3.metric("Max words", results["max_words"])
    c4.metric("Std deviation", results["std_words"])

    # Histogram
    st.subheader("Comment Length Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["word_count"], bins=30)
    ax.set_xlabel("Number of words")
    ax.set_ylabel("Number of comments")
    st.pyplot(fig)

    st.subheader("Top Words")
    st.write(results["top_words"])

    st.subheader("Strict Legislative Layer")
    st.metric("Article reference + Amendment verb (%)", results["strict_layer"])
    st.caption(
        "Percentage of comments that contain both a legislative article reference "
        "and a proposal for amendment (e.g. 'να προστεθεί', 'να διαγραφεί'). "
        "This approximates targeted legislative input."
    )

    st.subheader("Method & Configuration")
    st.write("Detected chapters:", chapters)
    st.write("Stopwords used:", stopwords_input)
    st.write("Policy keywords:", policy_keywords_input)
    st.write("Amendment verbs:", amendment_keywords_input)
    st.write("Run timestamp:", str(pd.Timestamp.now()))
