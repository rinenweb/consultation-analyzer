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

# =========================================================
# SESSION STATE
# =========================================================

if "abort" not in st.session_state:
    st.session_state.abort = False

if "results" not in st.session_state:
    st.session_state.results = None

# =========================================================
# LANGUAGE
# =========================================================

col_left, col_right = st.columns([10,1])
with col_right:
    lang_flag = st.radio("", ["🇬🇧", "🇬🇷"], horizontal=True, label_visibility="collapsed")

lang = "en" if lang_flag == "🇬🇧" else "el"
T = TRANSLATIONS[lang]

st.title(T["title"])
st.markdown(T["subtitle"])

url_input = st.text_input(T["input_label"])

# =========================================================
# ADVANCED SETTINGS
# =========================================================

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

BASE = "https://www.opengov.gr/minenv/"

# =========================================================
# SCRAPING
# =========================================================

def get_chapters(parent_id):
    r = requests.get(f"{BASE}?p={parent_id}")
    soup = BeautifulSoup(r.text, "html5lib")
    nav = soup.find("ul", class_="other_posts")
    chapters = []
    if nav:
        for a in nav.find_all("a", class_="list_comments_link"):
            match = re.search(r"\?p=(\d+)", a["href"])
            if match:
                chapters.append({
                    "pid": int(match.group(1)),
                    "title": a.get("title", "")
                })
    return chapters


def run_scraping(parent_id):

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
        status.write(f"{T['scraping_chapter']} {i+1}/{len(chapters)} (p={pid})")

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
                cid = li.get("id","").replace("comment-","")
                user_block = li.find("div", class_="user")
                if user_block:
                    user_block.extract()
                text = li.get_text("\n", strip=True)
                rows.append({
                    "chapter_p": pid,
                    "comment_id": cid,
                    "text": text
                })

            time.sleep(0.15)

        progress.progress((i+1)/len(chapters))

    progress.empty()
    status.empty()
    return pd.DataFrame(rows), chapters


@st.cache_data(ttl=600, show_spinner=False)
def run_scraping_cached(parent_id):
    return run_scraping(parent_id)

# =========================================================
# OPTIMIZED FUZZY
# =========================================================

def optimized_fuzzy_groups(df, threshold):

    texts = df["text_clean"].tolist()
    buckets = {}

    # bucket by first 40 characters
    for idx, text in enumerate(texts):
        key = text[:40]
        buckets.setdefault(key, []).append(idx)

    group_sizes = {}
    group_ids = {}

    for bucket in buckets.values():

        for i in range(len(bucket)):
            idx_i = bucket[i]
            if idx_i in group_sizes:
                continue

            t1 = texts[idx_i]
            group = [idx_i]

            for j in bucket[i+1:]:

                t2 = texts[j]

                # length ratio filter
                if abs(len(t1)-len(t2))/max(len(t1),len(t2)) > 0.2:
                    continue

                if SequenceMatcher(None, t1, t2).ratio() >= threshold:
                    group.append(j)

            if len(group) > 1:
                group_sizes[idx_i] = len(group)
                group_ids[idx_i] = df.loc[group,"comment_id"].astype(str).tolist()

    df["dup_size"] = 1
    for k,v in group_sizes.items():
        df.loc[k,"dup_size"] = v

    return group_sizes, group_ids

# =========================================================
# RUN
# =========================================================

if run_button and url_input:

    st.session_state.abort = False

    if "opengov.gr" not in url_input and not url_input.isdigit():
        st.error("Only opengov.gr consultations or valid parent IDs are allowed.")
        st.stop()

    parent_id = re.search(r"\?p=(\d+)", url_input).group(1) if "opengov.gr" in url_input else url_input

    col_spin, col_abort = st.columns([8,1])
    with col_abort:
        if st.button(T["abort"]):
            st.session_state.abort = True
            st.stop()

    start = time.time()

    with col_spin:
        with st.spinner(T["scraping"]):
            df, chapters = run_scraping_cached(parent_id)

    if time.time() - start < 0.5:
        st.info(T["loaded_cache"])

    if df.empty:
        st.warning("No comments found or scraping aborted.")
        st.stop()
    
    st.success(T["completed"])

    # ================= ANALYSIS =================

    df["text_clean"] = df["text"].str.strip().str.lower()
    df["word_count"] = df["text"].str.split().apply(len)

    if duplicate_method == T["exact_match"]:
        dup_counts = df["text_clean"].value_counts()
        df["dup_size"] = df["text_clean"].map(dup_counts)
        template_groups = dup_counts[dup_counts>1]
        template_ids = {}
    else:
        template_groups, template_ids = optimized_fuzzy_groups(
            df, similarity_threshold/100
        )
        template_groups = pd.Series(template_groups)

    campaign_share = round((df["dup_size"]>1).mean()*100,2)

    policy_patterns = [w.strip() for w in policy_keywords_input.split(",")]
    amend_patterns = [w.strip() for w in amendment_keywords_input.split(",")]

    df["mentions_article"] = df["text_clean"].str.contains("|".join(policy_patterns), regex=True)
    df["mentions_amendment"] = df["text_clean"].str.contains("|".join(amend_patterns), regex=True)

    strict_layer = round(
        (df["mentions_article"] & df["mentions_amendment"]).mean()*100,2
    )

    mean = df["word_count"].mean()
    median = df["word_count"].median()
    std = df["word_count"].std()

    st.session_state.results = {
        "df": df,
        "chapters": chapters,
        "campaign_share": campaign_share,
        "strict_layer": strict_layer,
        "mean": mean,
        "median": median,
        "std": std,
        "template_groups": template_groups,
        "template_ids": template_ids
    }

# =========================================================
# DISPLAY
# =========================================================

if st.session_state.results:

    R = st.session_state.results
    df = R["df"]

    col1, col2, col3 = st.columns(3)
    col1.metric(T["campaign"], R["campaign_share"], help=T["campaign_help"])
    col2.metric(T["strict"], R["strict_layer"], help=T["strict_desc"])
    col3.metric(T["mean"], round(R["mean"],2), help=T["mean_help"])

    # -------- Templates --------

    if len(R["template_groups"]) > 0:
        with st.expander(T["top_templates"]):

            top = R["template_groups"].sort_values(ascending=False).head(5)

            for idx,count in top.items():
                text = df.loc[int(idx),"text"]
                st.markdown(f"**{T['occurrences']}: {count}**")
                preview = text[:400] + ("..." if len(text)>400 else "")
                st.write(preview)

                with st.expander(T["show_full_text"]):
                    st.write(text)

                    ids = R["template_ids"].get(idx,[])[:10]
                    links = [f"[{cid}]({BASE}?c={cid})" for cid in ids]
                    st.markdown(" ".join(links))

                st.markdown("---")

    # -------- KDE --------

    st.subheader(T["distribution"])

    fig, ax = plt.subplots(figsize=(10,4))
    kde = gaussian_kde(df["word_count"])
    x = np.linspace(df["word_count"].min(),df["word_count"].max(),500)
    y = kde(x)

    ax.plot(x,y, label=T["density"])
    ax.axvline(R["mean"], linestyle="--", label=T["mean_line"])
    ax.axvline(R["median"], linestyle=":", label=T["median_line"])
    ax.set_xlabel(T["word_count_label"])
    ax.legend()

    st.pyplot(fig)

