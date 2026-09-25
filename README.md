# Consultation Analyzer

A Streamlit tool that scrapes and analyzes public consultations from **opengov.gr**, producing rule-based, methodologically transparent metrics: duplicate/campaign detection, legislative-relevance layers, comment-length statistics, and consultation-duration assessment.

Developed within the Postgraduate Programme *«e-Government»* of the University of the Aegean.

## Supports both opengov.gr systems

opengov.gr currently exists as two separate systems, and this app supports both:

- **Legacy site** — old consultations at `www.opengov.gr/<ministry>/?p=NNNN`, now served from `archive.opengov.gr/<ministry>/?p=NNNN` after a 301 redirect. The app accepts either host. Data is obtained by scraping the consultation's HTML.
- **New platform** — consultations at `opengov.gr/<org-slug>/deliberations/<slug>/`, backed by a public REST API (`https://opengov.gr/wp-json/opengov/v1`). Since a bare URL there can't be resolved to a specific consultation without querying the API, the app exposes a **search** UI (by organization and/or free text) instead of a link-paste field.

Pick the system via the radio button at the top of the app; the rest of the pipeline (duplicate detection, legislative layers, statistics, charts, exports) is identical regardless of source.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
streamlit run streamlit_app.py
```

This opens the app at `http://localhost:8501`.

## Tests

Install test dependencies (adds `pytest` on top of the runtime requirements) and run the suite:

```powershell
pip install -r requirements-dev.txt
pytest
```

The tests cover the pure text/URL/date-handling logic directly, and the HTML-scraping / REST-API-fetching functions with mocked HTTP responses — no network access or live opengov.gr data is required to run them.

## Project structure

| File | Responsibility |
|---|---|
| `streamlit_app.py` | UI, source routing (legacy vs. new platform), the analysis pipeline (normalize → duplicate detection → legislative layers → statistics), charts, and CSV/JSON export. |
| `analysis_utils.py` | Legacy-site HTML scraping (`requests` + `BeautifulSoup`), duplicate detection (exact and fuzzy/`rapidfuzz`), text canonicalization, and helpers shared with the new-platform client (`classify_duration`, `get_comment_url`). |
| `new_opengov_api.py` | Client for the new platform's public REST API — organization listing, search, and fetching a consultation's comments with the same `(df, chapters, timing_info)` shape the legacy scraper returns. |
| `translations.py` | English/Greek UI strings. |
| `tests/` | Pytest suite (see above). |

## Methodology notes

- Duplicate/campaign detection and the legislative-relevance layers are keyword- and similarity-threshold based, with all thresholds editable from the "Advanced Settings" panel in the UI — see the in-app tooltips for exact definitions.
- Consultation-duration assessment compares the announced posting date against the closing date (from the sidebar on the legacy site, from `publish_date`/`expiry_date` on the new platform) against fixed benchmarks (under 14 days = insufficient, 14–20 = borderline, 21+ = satisfactory).
- On the new platform, comments are attributed to the specific article/section they were posted under via the API's `article_unique_id`/`article_title` fields, so the per-chapter breakdown works the same way as on the legacy site.

## Source

<https://github.com/rinenweb/consultation-analyzer/>
