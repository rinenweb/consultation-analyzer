import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

from analysis_utils import (
    build_comment_link,
    canonicalize_text,
    extract_base_and_parent,
    optimized_fuzzy_groups,
    render_steps,
    scrape_consultation_with_progress,
)
from translations import TRANSLATIONS

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
        "άρθρ,συνταγμ,οδηγ,ευρωπαϊκ,εε,ενωσιακ,παράγραφ,εδάφ,περίπτωσ,διάταξ,διατύπωσ,νομοτεχν",
        help=T["policy_help"]
    )
    legislative_logic = st.selectbox(
        T["legislative_logic_label"],
        [T["logic_and"], T["logic_or"]],
        index=0,
        help=T["legislative_logic_help"]
    )
    st.caption(T["logic_and_desc"] if legislative_logic == T["logic_and"] else T["logic_or_desc"])
    
    amendment_keywords_input = st.text_area(
        T["amend"],
        "να προστεθ,να διαγραφ,να αντικατασταθ,να τροποποιηθ,να απαλειφθ,να αναδιατυπωθ,να συμπληρωθ,να διευκρινιστ,να εξειδικευτ,να διορθωθ,να προβλεφθ,",
        help=T["amend_help"]
    )

    duplicate_method = st.radio(
        T["duplicate_method_label"],
        [T["fuzzy_match"], T["exact_match"]],
        horizontal=True,
        index=0
    )

    similarity_threshold = 80
    if duplicate_method == T["fuzzy_match"]:
        similarity_threshold = st.slider(
            T["similarity_threshold"],
            80, 100, 90,
            help=T["similarity_threshold_help"]
        )
        if similarity_threshold == 100:
            st.info(T["fuzzy_100_hint"])

run_button = st.button(T["run"])

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
    if st.session_state.running:
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
        T["step_scrape"],
        T["step_normalize"],
        T["step_duplicates"],
        T["step_strict"],
        T["step_stats"],
        T["step_render"],
    ]

    steps_ph = st.empty()
    done = set()
    active = 0
    render_steps(steps_ph, steps, active, done)

    # ================= STEP 1: SCRAPE =================

    df, chapters, timing_info = scrape_consultation_with_progress(parent_id, base, T)

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

    # ---- NEW: token-based bucketing for fuzzy pruning ----
    df["token_count"] = df["text_clean"].str.split().str.len()
    df["bucket"] = df["token_count"] // 10

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
            similarity_threshold,
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

    if legislative_logic == T["logic_or"]:
        strict_mask = df["mentions_article"] | df["mentions_amendment"]
    else:
        strict_mask = df["mentions_article"] & df["mentions_amendment"]
    
    strict_layer = round(strict_mask.mean() * 100, 2)
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
        "legislative_logic": legislative_logic,
        "strict_layer": strict_layer,
        "mean": mean,
        "median": median,
        "max": max_words,
        "std": std,
        "template_groups": template_groups,
        "template_ids": template_ids,
        "timing_info": timing_info,
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

if st.session_state.results and not st.session_state.running:
    R = st.session_state.results
    df = R["df"]
    base = R["base"]

    chapters_df = pd.DataFrame(R.get("chapters", []))

    chapter_title_map = {}
    if not chapters_df.empty and "pid" in chapters_df.columns:
        chapter_title_map = dict(zip(chapters_df["pid"], chapters_df["title"]))

    if R.get("legislative_logic") == T["logic_or"]:
        layer_metric_label = T.get("broad", "Broad Legislative Relevance Layer (%)")
        layer_metric_desc = T.get(
            "broad_desc",
            "Percentage of comments containing either a policy/article reference or an explicit amendment proposal."
        )
        layer_section_title = T.get("broad_layer_title", "Broad Legislative Relevance Layer")
    else:
        layer_metric_label = T.get("strict", "Strict Legislative Layer (%)")
        layer_metric_desc = T.get(
            "strict_desc",
            "Percentage of comments containing both an article reference and an explicit amendment proposal."
        )
        layer_section_title = T.get("targeted_layer_title", "Targeted Legislative Intervention Layer")
    
    # Strict intervention layer
    if R.get("legislative_logic") == T["logic_or"]:
        strict_mask = df.get("mentions_article", False) | df.get("mentions_amendment", False)
    else:
        strict_mask = df.get("mentions_article", False) & df.get("mentions_amendment", False)
    
    targeted_df = df.loc[strict_mask].copy()
    
    if not targeted_df.empty:
    
        targeted_df["chapter_title"] = targeted_df["chapter_p"].map(chapter_title_map).fillna("")
        
        targeted_df["comment_url"] = targeted_df["comment_id"].astype(str).apply(
            lambda cid: build_comment_link(base, cid)
        )
    
        targeted_df = targeted_df.sort_values(
            "word_count",
            ascending=False
        )
    
        targeted_records = targeted_df.to_dict("records")
    
    else:
        targeted_records = []

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
        layer_metric_label,
        R["strict_layer"],
        help=layer_metric_desc
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
                
                percentage = (count_int / len(df)) * 100
                st.markdown(
                    f"**{T.get('occurrences','Occurrences')}: {count_int} "
                    f"({percentage:.2f}% {T.get('of_total','of total comments')})**"
                )

                preview_len = 400
                preview = full_text[:preview_len]
                rest = full_text[preview_len:]
                
                if rest:
                    st.markdown(f"> {preview}...")
                else:
                    st.write(full_text)
                
                if rest:
                    with st.expander(T.get("show_full_text", "Show full template")):
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

    with st.expander(layer_section_title, expanded=False):
        
            if not targeted_records:
                st.info(
                    T.get(
                        "no_targeted",
                        "No targeted legislative intervention comments detected."
                    )
                )
        
            else:
        
                st.caption(
                    f"{len(targeted_records)} "
                    + T.get("targeted_comments_detected", "targeted comments detected")
                )
        
                for rec in targeted_records[:20]:
        
                    chapter_title = rec.get("chapter_title", "")
                    chapter_pid = rec.get("chapter_p", "")
                    comment_url = rec.get("comment_url")
                    text = rec.get("text", "")
        
                    header = []
        
                    if chapter_title:
                        header.append(f"**{T.get('chapter','Chapter')}:** {chapter_title}")
                    else:
                        header.append(f"**{T.get('chapter','Chapter')}:** {T.get('chapter_id','p')}={chapter_pid}")
        
                    st.markdown(" • ".join(header))
        
                    preview = text[:500] + ("..." if len(text) > 500 else "")
                    st.write(preview)
        
                    if comment_url:
                        st.markdown(
                            f"[{T.get('open_comment','Open original comment')}]({comment_url})"
                        )

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

    timing = R.get("timing_info", {}) or {}
duration_days = timing.get("duration_days")
duration_label = timing.get("duration_label")
duration_color = timing.get("duration_color")
posted_raw = timing.get("posted_raw")
closes_raw = timing.get("closes_raw")

if duration_days is not None:
    bg_map = {
        "red": "#fde8e8",
        "orange": "#fff4d6",
        "green": "#e6f4ea",
    }
    border_map = {
        "red": "#d93025",
        "orange": "#f9ab00",
        "green": "#188038",
    }
    text_map = {
        "red": "#8b1e1e",
        "orange": "#8a5a00",
        "green": "#0f5c2e",
    }

    bg_color = bg_map.get(duration_color, "#f3f4f6")
    border_color = border_map.get(duration_color, "#9ca3af")
    text_color = text_map.get(duration_color, "#374151")

    duration_display = (
        f"{int(duration_days)}"
        if float(duration_days).is_integer()
        else f"{duration_days:.2f}"
    )

    tooltip_text = T.get("duration_tooltip", "").format(
        posted=posted_raw or "-",
        closes=closes_raw or "-",
    )

    st.markdown(
        f"""
        <div style="
            margin-top: 1rem;
            padding: 0.9rem 1rem;
            border-left: 6px solid {border_color};
            background: {bg_color};
            border-radius: 0.5rem;
            color: {text_color};
        " title="{tooltip_text}">
            <div style="font-weight: 700; margin-bottom: 0.25rem;">
                {T.get("duration_block_title", "Consultation duration")}
            </div>
            <div>
                {T.get("duration_block_text", "This consultation remained open for {days} days.").format(days=duration_display)}
            </div>
            <div style="margin-top: 0.35rem; font-size: 0.95rem;">
                <strong>{T.get("duration_assessment_label", "Assessment")}:</strong> {duration_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(tooltip_text)
    
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
    if R.get("legislative_logic") == T["logic_or"]:
        export_df["strict_flag"] = export_df.get("mentions_article", False) | export_df.get("mentions_amendment", False)
    else:
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
        "legislative_logic": R.get("legislative_logic"),
        "strict_layer_pct": float(R.get("strict_layer")),
        "policy_keywords": R.get("policy_keywords"),
        "amendment_keywords": R.get("amendment_keywords"),
        "consultation_timing": {
            "posted_raw": timing.get("posted_raw"),
            "closes_raw": timing.get("closes_raw"),
            "posted_dt": timing.get("posted_dt"),
            "closes_dt": timing.get("closes_dt"),
            "duration_days": timing.get("duration_days"),
            "duration_label": timing.get("duration_label"),
        },
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
        st.write(T.get("legislative_logic_label","Legislative matching logic") + ":", R.get("legislative_logic", T["logic_and"]))
        st.write(T.get("amend","Amendment verbs") + ":", R["amendment_keywords"])
        st.write(T.get("timestamp","Run timestamp") + ":", R["timestamp"])
        timing = R.get("timing_info", {}) or {}
        if timing.get("duration_days") is not None:
            st.write(
                T.get("duration_block_title", "Consultation duration") + ":",
                f"{timing.get('duration_days')} {T.get('days_unit', 'days')}"
            )

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
