from datetime import datetime
from unittest.mock import MagicMock

import pytest

import new_opengov_api
from new_opengov_api import (
    _parse_api_datetime,
    _strip_html,
    fetch_new_consultation_with_progress,
    search_deliberations,
)
from tests.conftest import FakeResponse


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

def test_strip_html_removes_tags_and_decodes_entities():
    assert _strip_html("<p>Hello &amp; welcome</p>\n") == "Hello & welcome"


@pytest.mark.parametrize("value", [None, ""])
def test_strip_html_empty(value):
    assert _strip_html(value) == ""


# ---------------------------------------------------------------------------
# _parse_api_datetime
# ---------------------------------------------------------------------------

def test_parse_api_datetime_valid():
    assert _parse_api_datetime("2026-07-14 18:21:02") == datetime(2026, 7, 14, 18, 21, 2)


@pytest.mark.parametrize("value", [None, "", "14/07/2026", "not a date"])
def test_parse_api_datetime_invalid(value):
    assert _parse_api_datetime(value) is None


# ---------------------------------------------------------------------------
# search_deliberations: verify which endpoint gets called for each combination
# ---------------------------------------------------------------------------

def test_search_deliberations_with_query_hits_search_endpoint(monkeypatch):
    mock_get = MagicMock(return_value=FakeResponse(json_data=[{"id": 1}]))
    monkeypatch.setattr(new_opengov_api.requests, "get", mock_get)

    result = search_deliberations("νόμου", site_id=51)

    assert result == [{"id": 1}]
    called_url = mock_get.call_args.args[0]
    called_params = mock_get.call_args.kwargs["params"]
    assert called_url.endswith("/search")
    assert called_params["q"] == "νόμου"
    assert called_params["site_id"] == 51


def test_search_deliberations_empty_query_with_site_id_hits_subsite_posts(monkeypatch):
    mock_get = MagicMock(return_value=FakeResponse(json_data=[]))
    monkeypatch.setattr(new_opengov_api.requests, "get", mock_get)

    search_deliberations("", site_id=42)

    called_url = mock_get.call_args.args[0]
    assert called_url.endswith("/subsite-posts")


def test_search_deliberations_empty_query_no_site_id_hits_all_posts(monkeypatch):
    mock_get = MagicMock(return_value=FakeResponse(json_data=[]))
    monkeypatch.setattr(new_opengov_api.requests, "get", mock_get)

    search_deliberations("   ", site_id=None)

    called_url = mock_get.call_args.args[0]
    assert called_url.endswith("/all-posts")


# ---------------------------------------------------------------------------
# fetch_new_consultation_with_progress
# ---------------------------------------------------------------------------

FAKE_POST = {
    "id": 123,
    "title": "Test Consultation",
    "acf_fields": {
        "publish_date": "2026-01-01 10:00:00",
        "expiry_date": "2026-01-20 10:00:00",
    },
}

FAKE_COMMENTS_PAGE_1 = [
    {
        "id": "1",
        "article_unique_id": "article_aaa",
        "article_title": "Άρθρο 1",
        "content": "<p>Πρώτο σχόλιο</p>",
        "comment_url": "https://opengov.gr/comment-view/?cid=1&sid=1",
        "date": "2026-01-05 09:00:00",
    },
    {
        "id": "2",
        "article_unique_id": "article_bbb",
        "article_title": "",
        "content": "<p>Δεύτερο σχόλιο</p>",
        "comment_url": "https://opengov.gr/comment-view/?cid=2&sid=1",
        "date": "2026-01-06 09:00:00",
    },
]


def test_fetch_new_consultation_builds_expected_shape(fake_st, monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url.endswith("/comments"):
            return FakeResponse(json_data=FAKE_COMMENTS_PAGE_1, headers={"X-WP-TotalPages": "1"})
        return FakeResponse(json_data=FAKE_POST)

    session_mock = MagicMock()
    session_mock.get.side_effect = fake_get
    monkeypatch.setattr(new_opengov_api.requests, "Session", lambda: session_mock)

    df, chapters, timing_info = fetch_new_consultation_with_progress(123, 1, {})

    assert len(df) == 2
    assert set(df["chapter_p"]) == {"article_aaa", "article_bbb"}
    assert df.loc[df["comment_id"] == "1", "text"].iloc[0] == "Πρώτο σχόλιο"

    # article_title falls back to the post title when the API gives an empty one
    chapter_by_id = {c["pid"]: c["title"] for c in chapters}
    assert chapter_by_id["article_aaa"] == "Άρθρο 1"
    assert chapter_by_id["article_bbb"] == "Test Consultation"

    assert timing_info["duration_days"] == 19.0
    assert timing_info["duration_color"] == "orange"


def test_fetch_new_consultation_stops_when_aborted(fake_st, monkeypatch):
    fake_st.abort = True

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/comments"):
            pytest.fail("should not fetch comments once aborted")
        return FakeResponse(json_data=FAKE_POST)

    session_mock = MagicMock()
    session_mock.get.side_effect = fake_get
    monkeypatch.setattr(new_opengov_api.requests, "Session", lambda: session_mock)

    df, chapters, timing_info = fetch_new_consultation_with_progress(123, 1, {})

    assert df.empty
    assert chapters == []
