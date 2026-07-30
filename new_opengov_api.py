from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from analysis_utils import classify_duration

NEW_API_BASE = "https://opengov.gr/wp-json/opengov/v1"
REQUEST_TIMEOUT = 25


def _get(path: str, params: dict = None):
    r = requests.get(f"{NEW_API_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r


@st.cache_data(ttl=3600)
def list_organizations():
    return _get("/organizations").json()


def search_deliberations(query: str, site_id=None, per_page: int = 20):
    """
    Wraps /search. With an empty query, falls back to browsing everything
    (only a handful of consultations exist on the new platform right now),
    so the search UI never comes back empty during quiet periods.
    """
    query = (query or "").strip()
    params = {"per_page": per_page}
    if site_id:
        params["site_id"] = site_id

    if query:
        params["q"] = query
        return _get("/search", params=params).json()

    if site_id:
        return _get("/subsite-posts", params=params).json()
    return _get("/all-posts", params=params).json()


def _strip_html(html_text: str) -> str:
    if not html_text:
        return ""
    return BeautifulSoup(html_text, "html5lib").get_text("\n", strip=True)


def _parse_api_datetime(value: str):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def fetch_new_consultation_with_progress(post_id, site_id, translations: dict):
    """
    Same contract as scrape_consultation_with_progress: returns
    (df, chapters, timing_info) so the rest of the pipeline in
    streamlit_app.py doesn't need to branch on the data source.
    """
    session = requests.Session()

    post = session.get(
        f"{NEW_API_BASE}/post/{post_id}",
        params={"site_id": site_id},
        timeout=REQUEST_TIMEOUT,
    ).json()
    post_title = post.get("title", "")
    acf = post.get("acf_fields", {}) or {}

    posted_dt = _parse_api_datetime(acf.get("publish_date"))
    closes_dt = _parse_api_datetime(acf.get("expiry_date"))
    duration_days = None
    if posted_dt and closes_dt:
        duration_days = round((closes_dt - posted_dt).total_seconds() / 86400, 2)
    duration_label, duration_color = classify_duration(duration_days, translations)

    timing_info = {
        "posted_raw": acf.get("publish_date"),
        "closes_raw": acf.get("expiry_date"),
        "posted_dt": posted_dt.isoformat() if posted_dt else None,
        "closes_dt": closes_dt.isoformat() if closes_dt else None,
        "duration_days": duration_days,
        "duration_label": duration_label,
        "duration_color": duration_color,
    }

    prog = st.progress(0.0)
    status = st.empty()

    all_rows = []
    chapter_titles = {}
    page = 1
    total_pages = 1

    while page <= total_pages:
        if st.session_state.abort:
            break

        status.write(f"{translations.get('scraping_chapter', 'Scraping chapter')} {page}/{total_pages}")

        r = session.get(
            f"{NEW_API_BASE}/post/{post_id}/comments",
            params={"site_id": site_id, "per_page": 100, "page": page},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        comments = r.json()
        total_pages = int(r.headers.get("X-WP-TotalPages", "1"))

        for c in comments:
            article_id = c.get("article_unique_id") or str(post_id)
            article_title = c.get("article_title") or post_title
            chapter_titles.setdefault(article_id, article_title)

            all_rows.append({
                "chapter_p": article_id,
                "comment_id": c.get("id"),
                "text": _strip_html(c.get("content")),
                "comment_url": c.get("comment_url"),
                "comment_date": _parse_api_datetime(c.get("date")),
            })

        prog.progress(min(page / total_pages, 1.0))
        page += 1

    prog.empty()
    status.empty()

    chapters = [{"pid": aid, "title": title} for aid, title in chapter_titles.items()]

    return pd.DataFrame(all_rows), chapters, timing_info
