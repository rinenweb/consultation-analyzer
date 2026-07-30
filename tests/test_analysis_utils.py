from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from analysis_utils import (
    build_comment_link,
    canonicalize_text,
    classify_duration,
    extract_base_and_parent,
    get_chapters,
    get_comment_url,
    optimized_fuzzy_groups,
    parse_greek_datetime,
    scrape_chapter,
    scrape_consultation_timing,
)
from tests.conftest import FakeResponse


# ---------------------------------------------------------------------------
# extract_base_and_parent
# ---------------------------------------------------------------------------

def test_extract_base_and_parent_www():
    base, parent_id = extract_base_and_parent("https://www.opengov.gr/immigration/?p=2000")
    assert base == "https://www.opengov.gr/immigration/"
    assert parent_id == "2000"


def test_extract_base_and_parent_archive():
    base, parent_id = extract_base_and_parent("https://archive.opengov.gr/digitalandbrief/?p=3832")
    assert base == "https://archive.opengov.gr/digitalandbrief/"
    assert parent_id == "3832"


@pytest.mark.parametrize("url", [
    "",
    None,
    "not a url",
    "https://opengov.gr/minocp/deliberations/some-slug/",  # new platform, not this parser's job
    "https://evil.example.com/www.opengov.gr/x/?p=1",
])
def test_extract_base_and_parent_invalid(url):
    assert extract_base_and_parent(url) == (None, None)


# ---------------------------------------------------------------------------
# comment URL helpers
# ---------------------------------------------------------------------------

def test_build_comment_link():
    assert build_comment_link("https://www.opengov.gr/x/", "42") == "https://www.opengov.gr/x/?c=42"


def test_get_comment_url_prefers_provided_url():
    url = get_comment_url("42", "https://www.opengov.gr/x/", "https://opengov.gr/comment-view/?cid=42&sid=1")
    assert url == "https://opengov.gr/comment-view/?cid=42&sid=1"


def test_get_comment_url_falls_back_to_legacy_pattern():
    url = get_comment_url("42", "https://www.opengov.gr/x/", None)
    assert url == "https://www.opengov.gr/x/?c=42"


# ---------------------------------------------------------------------------
# classify_duration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("days,expected_color", [
    (0, "red"),
    (13.99, "red"),
    (14, "orange"),
    (20.99, "orange"),
    (21, "green"),
    (100, "green"),
])
def test_classify_duration_thresholds(days, expected_color):
    label, color = classify_duration(days, {})
    assert color == expected_color
    assert label  # some non-empty fallback label was returned


def test_classify_duration_none():
    assert classify_duration(None, {}) == (None, None)


def test_classify_duration_uses_provided_translations():
    label, color = classify_duration(5, {"duration_insufficient": "Ανεπαρκής διάρκεια"})
    assert label == "Ανεπαρκής διάρκεια"
    assert color == "red"


# ---------------------------------------------------------------------------
# parse_greek_datetime
# ---------------------------------------------------------------------------

def test_parse_greek_datetime_valid():
    dt = parse_greek_datetime("2 Μαρτίου 2026, 15:30")
    assert dt == datetime(2026, 3, 2, 15, 30)


def test_parse_greek_datetime_unknown_month():
    assert parse_greek_datetime("2 Ξεναρίου 2026, 15:30") is None


@pytest.mark.parametrize("value", [None, "", "garbage text"])
def test_parse_greek_datetime_invalid(value):
    assert parse_greek_datetime(value) is None


# ---------------------------------------------------------------------------
# canonicalize_text
# ---------------------------------------------------------------------------

def test_canonicalize_text_strips_accents_and_punctuation():
    assert canonicalize_text("Άρθρο 5, παράγραφος 2!") == "αρθρο 5 παραγραφος 2"


def test_canonicalize_text_collapses_whitespace():
    assert canonicalize_text("  πολλά   κενά   εδώ  ") == "πολλα κενα εδω"


def test_canonicalize_text_empty():
    assert canonicalize_text("") == ""
    assert canonicalize_text(None) == ""


# ---------------------------------------------------------------------------
# legacy HTML scraping (mocked requests.Session, real BeautifulSoup parsing)
# ---------------------------------------------------------------------------

TIMING_HTML = """
<div class="sidespot red_spot">
    <h4>Αναρτήθηκε<br/>
    <span>2 Μαρτίου 2026, 15:30</span><br/>
    Ανοικτή σε Σχόλια έως<br/>
    <span>16 Μαρτίου 2026, 23:59</span>
    </h4>
</div>
"""

CHAPTERS_HTML = """
<ul class="other_posts">
  <li><a class="list_comments_link" href="https://www.opengov.gr/test/?p=101" title="Άρθρο 1">Άρθρο 1</a></li>
  <li><a class="list_comments_link" href="https://www.opengov.gr/test/?p=102" title="Άρθρο 2">Άρθρο 2</a></li>
</ul>
"""

COMMENTS_PAGE_HTML = """
<ul class="comment_list">
  <li class="comment" id="comment-501">
    <div class="user">Some Author</div>
    Comment text one here.
  </li>
  <li class="comment" id="comment-502">
    <div class="user">Another Author</div>
    Comment text two here.
  </li>
</ul>
"""

EMPTY_COMMENTS_HTML = "<div>no comments here</div>"


def test_scrape_consultation_timing_parses_duration():
    session = MagicMock()
    session.get.return_value = FakeResponse(text=TIMING_HTML)

    result = scrape_consultation_timing("101", "https://www.opengov.gr/test/", session, {})

    assert result["posted_raw"] == "2 Μαρτίου 2026, 15:30"
    assert result["closes_raw"] == "16 Μαρτίου 2026, 23:59"
    assert result["duration_days"] == 14.35
    assert result["duration_color"] == "orange"


def test_get_chapters_parses_nav():
    session = MagicMock()
    session.get.return_value = FakeResponse(text=CHAPTERS_HTML)

    chapters = get_chapters("100", "https://www.opengov.gr/test/", session)

    assert chapters == [
        {"pid": 101, "title": "Άρθρο 1"},
        {"pid": 102, "title": "Άρθρο 2"},
    ]


def test_scrape_chapter_extracts_rows_and_strips_user_block(fake_st):
    session = MagicMock()
    session.get.side_effect = [
        FakeResponse(text=COMMENTS_PAGE_HTML),
        FakeResponse(text=EMPTY_COMMENTS_HTML),
    ]

    rows = scrape_chapter(101, "https://www.opengov.gr/test/", session)

    assert rows == [
        {"chapter_p": 101, "comment_id": "501", "text": "Comment text one here."},
        {"chapter_p": 101, "comment_id": "502", "text": "Comment text two here."},
    ]


def test_scrape_chapter_stops_immediately_when_aborted(fake_st):
    fake_st.abort = True
    session = MagicMock()

    rows = scrape_chapter(101, "https://www.opengov.gr/test/", session)

    assert rows == []
    session.get.assert_not_called()


# ---------------------------------------------------------------------------
# optimized_fuzzy_groups
# ---------------------------------------------------------------------------

def _bucketed_df(texts):
    df = pd.DataFrame({
        "comment_id": [str(i) for i in range(len(texts))],
        "text_clean": texts,
    })
    df["token_count"] = df["text_clean"].str.split().str.len()
    df["bucket"] = df["token_count"] // 10
    return df


def test_optimized_fuzzy_groups_finds_near_duplicates(fake_st):
    texts = [
        "αυτο ειναι ενα σχολιο υποστηριξης του αρθρου πεντε",
        "αυτο ειναι ενα σχολιο υποστηριξης του αρθρου πεντε ναι",
        "εντελως διαφορετικο σχολιο για αλλο θεμα τελειως ασχετο",
    ]
    df = _bucketed_df(texts)

    group_sizes, group_ids = optimized_fuzzy_groups(df, threshold=80)

    assert len(group_sizes) == 1
    (representative, size), = group_sizes.items()
    assert size == 2
    assert set(group_ids[representative]) == {"0", "1"}
    assert df.loc[2, "dup_size"] == 1


def test_optimized_fuzzy_groups_aborts_early(fake_st):
    fake_st.abort = True
    df = _bucketed_df(["κειμενο ενα", "κειμενο δυο"])

    group_sizes, group_ids = optimized_fuzzy_groups(df, threshold=80)

    assert group_sizes == {}
    assert group_ids == {}
