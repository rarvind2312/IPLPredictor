"""
DEPRECATED: Automatic IPL match URL discovery and pre-prediction scorecard scraping lived here.

That pipeline (IPL yearly results pages, team schedule crawling, Cricbuzz/Cricinfo series pages)
has been **removed**. Historical data is loaded in the **ingest** stage into SQLite (not during prediction).

This module is kept as a placeholder under ``archive/source_deprecated/providers/`` (nothing in the app imports it).
Add new provider code only when needed for non-scraping workflows.
"""

from __future__ import annotations

# Intentionally empty — former exports (e.g. ``fetch_recent_match_urls_for_team``) removed.
