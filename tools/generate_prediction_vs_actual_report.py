"""
Build docs/prediction_vs_actual_report.md from live scorecard + predictor audit.

Uses the same pipeline as tools/validate_last_ipl_2026_matches.run_audit (no prediction logic changes).

Run from repo root (network required for IPLT20 fetch + squads):

  PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python tools/generate_prediction_vs_actual_report.py
  PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python tools/generate_prediction_vs_actual_report.py --urls URL1 URL2

Optional:

  --output PATH   default: docs/prediction_vs_actual_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.validate_last_ipl_2026_matches import DEFAULT_MATCH_URLS, run_audit


def load_match_payloads_from_sqlite(limit: int) -> dict[str, dict[str, Any]]:
    """
    Use ingested ``match_results.raw_payload`` (actual scorecard JSON) when live IPL HTML
    no longer exposes teams in the parser.
    """
    import db
    import ipl_teams

    db.init_schema()
    out: dict[str, dict[str, Any]] = {}
    want = max(int(limit), 1)
    # Over-fetch: recent rows may be non-IPL (e.g. PSL) in a shared DB; keep scanning until we have `want` IPL fixtures.
    fetch_cap = max(want * 25, want)
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT url, team_a, team_b, raw_payload
            FROM match_results
            WHERE raw_payload IS NOT NULL AND trim(raw_payload) != ''
              AND team_a IS NOT NULL AND trim(team_a) != ''
              AND team_b IS NOT NULL AND trim(team_b) != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (fetch_cap,),
        ).fetchall()
    for r in rows:
        if len(out) >= want:
            break
        url = str(r["url"] or "").strip()
        if not url:
            continue
        try:
            payload = json.loads(r["raw_payload"])
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        teams_raw = [str(t or "").strip() for t in (payload.get("teams") or []) if str(t or "").strip()]
        if len(teams_raw) < 2:
            ta = str(r["team_a"] or "").strip()
            tb = str(r["team_b"] or "").strip()
            if ta and tb:
                payload["teams"] = [ta, tb]
                teams_raw = [ta, tb]
        if len(teams_raw) < 2:
            continue
        la = ipl_teams.franchise_label_for_storage(teams_raw[0])
        lb = ipl_teams.franchise_label_for_storage(teams_raw[1])
        if not ipl_teams.slug_for_canonical_label(la) or not ipl_teams.slug_for_canonical_label(lb):
            continue
        out[url] = payload
    return out


def _fmt_list(xs: list[str]) -> str:
    if not xs:
        return "—"
    return ", ".join(str(x) for x in xs)


def _md_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def build_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Prediction vs actual XI — comparison report")
    lines.append("")
    lines.append("**Generated:** " + datetime.now(timezone.utc).isoformat())
    lines.append("")
    lines.append("**Data sources:** `parsers.router.parse_scorecard` (IPLT20 match URLs), ")
    lines.append("`squad_fetch.fetch_squad_for_slug` (current official squads), `predictor.run_prediction`.")
    lines.append("")
    lines.append("**Caveats:** Squads are **current** IPLT20 listings; XIs for past matches may include ")
    lines.append("players not on today’s page. Impact Player **actual** selections are **not** parsed from ")
    lines.append("scorecards in this pipeline — only **model-predicted** impact order is shown.")
    lines.append("")
    lines.append(
        "**XI / bowling comparison keys:** Overlap, missed/extra, and bowling-usage sets use "
        "``player_registry.audit_player_identity_key`` (aliases / history keys), then "
        "``learner.normalize_player_key`` fallback — audit layer only; prediction unchanged."
    )
    lines.append("")
    src_note = audit.get("payload_source_note") or ""
    if src_note:
        lines.append(f"**Run mode:** {src_note}")
        lines.append("")
    lines.append("---")
    lines.append("")

    per = audit.get("per_match") or []
    summary = audit.get("summary") or {}

    # Aggregate for sections A–D
    gap_tag_counter: Counter[str] = Counter()
    player_miss_counter: Counter[str] = Counter()
    player_extra_counter: Counter[str] = Counter()
    team_overlap_sum: dict[str, list[int]] = defaultdict(list)
    low_overlap_rows: list[tuple[str, str, int, str]] = []  # team, url, overlap, venue

    for m in per:
        if m.get("error"):
            continue
        url = str(m.get("url") or "")
        meta = m.get("meta") or {}
        venue = str(meta.get("venue") or "")
        avp = m.get("actual_vs_predicted") or {}
        for team_label, block in avp.items():
            if not isinstance(block, dict):
                continue
            cp = int(block.get("correct_picks") or 0)
            team_overlap_sum[team_label].append(cp)
            if cp < 8:
                low_overlap_rows.append((team_label, url, cp, venue))
            for t in block.get("likely_gap_tags") or []:
                gap_tag_counter[str(t)] += 1
            for nm in block.get("missed_actual") or []:
                player_miss_counter[str(nm)] += 1
            for nm in block.get("extra_predicted") or []:
                player_extra_counter[str(nm)] += 1

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Matches requested: **{summary.get('matches_requested', 0)}**")
    lines.append(f"- Matches audited (parsed + predicted): **{summary.get('matches_audited', 0)}**")
    failed = summary.get("matches_failed") or []
    if failed:
        lines.append(f"- **Failed:** {len(failed)} (fetch/parse/prediction)")
        for fm in failed[:8]:
            lines.append(f"  - `{fm.get('url')}` — `{fm.get('error')}`")
    lines.append("")

    lines.append("## A. Top recurring mismatch patterns (heuristic tags)")
    lines.append("")
    if gap_tag_counter:
        lines.append(_md_table_row(["Tag", "Count"]))
        lines.append(_md_table_row(["---", "---"]))
        for tag, c in gap_tag_counter.most_common(15):
            lines.append(_md_table_row([tag, str(c)]))
    else:
        lines.append("— No gap tags recorded (or no successful matches).")
    lines.append("")

    lines.append("## B. Players frequently mispredicted")
    lines.append("")
    lines.append("### Most often **missed** (in actual XI, not in predicted XI)")
    lines.append("")
    if player_miss_counter:
        lines.append(_md_table_row(["Player", "Miss count"]))
        lines.append(_md_table_row(["---", "---"]))
        for nm, c in player_miss_counter.most_common(20):
            lines.append(_md_table_row([nm, str(c)]))
    else:
        lines.append("—")
    lines.append("")
    lines.append("### Most often **false positives** (predicted XI, not in actual XI)")
    lines.append("")
    if player_extra_counter:
        lines.append(_md_table_row(["Player", "Extra count"]))
        lines.append(_md_table_row(["---", "---"]))
        for nm, c in player_extra_counter.most_common(20):
            lines.append(_md_table_row([nm, str(c)]))
    else:
        lines.append("—")
    lines.append("")

    lines.append("## C. Teams where XI overlap is lowest (this run)")
    lines.append("")
    rows_t = []
    for team, vals in team_overlap_sum.items():
        if vals:
            avg = sum(vals) / len(vals)
            rows_t.append((team, min(vals), sum(vals) / len(vals), len(vals)))
    rows_t.sort(key=lambda x: x[2])
    if rows_t:
        lines.append(_md_table_row(["Team (canonical)", "Min overlap /11", "Mean overlap /11", "Innings"]))
        lines.append(_md_table_row(["---", "---", "---", "---"]))
        for team, mn, mean, n in rows_t:
            lines.append(_md_table_row([team, str(mn), f"{mean:.2f}", str(n)]))
    else:
        lines.append("—")
    lines.append("")

    lines.append("## D. Conditions where the model struggled (low overlap, this run)")
    lines.append("")
    lines.append("Teams with **correct_picks < 8** (arbitrary threshold for this report):")
    lines.append("")
    if low_overlap_rows:
        lines.append(_md_table_row(["Team", "Overlap", "Venue (scorecard)", "URL"]))
        lines.append(_md_table_row(["---", "---", "---", "---"]))
        for team, url, cp, venue in sorted(low_overlap_rows, key=lambda x: x[2])[:25]:
            lines.append(_md_table_row([team, str(cp), venue[:60], url]))
    else:
        lines.append("— None under threshold, or no data.")
    lines.append("")
    lines.append("For each match, see **Conditions used** (venue, weather snapshot, toss unknown) in the per-match section.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Per-match detail")
    lines.append("")

    for i, m in enumerate(per, 1):
        lines.append(f"### Match {i}")
        lines.append("")
        if m.get("error"):
            lines.append(f"- **Error:** `{m.get('error')}`")
            if m.get("detail"):
                lines.append(f"- **Detail:** {m.get('detail')}")
            stc_err = m.get("squad_temporal_confound") or {}
            if isinstance(stc_err, dict) and stc_err.get("roster_drift_likely"):
                lines.append(
                    f"- **Squad vs scorecard season:** match year **{stc_err.get('match_date_year', '—')}** vs "
                    f"squad snapshot year **{stc_err.get('squad_snapshot_assumed_calendar_year', '—')}** — "
                    f"_{stc_err.get('note', '')}_"
                )
            if m.get("team_a") is not None or m.get("team_b") is not None:
                lines.append(f"- **Team labels:** `{m.get('team_a')}` vs `{m.get('team_b')}`")
            axi = m.get("actual_xi_from_scorecard")
            if isinstance(axi, dict) and axi:
                lines.append("- **Actual XI (scorecard only — prediction did not complete):**")
                for tl, names in sorted(axi.items()):
                    lines.append(f"  - **{tl}:** {_fmt_list(list(names or []))}")
            lines.append(f"- URL: `{m.get('url')}`")
            lines.append("")
            continue

        lines.append(f"- **URL:** `{m.get('url')}`")
        if m.get("payload_source"):
            lines.append(f"- **Payload source:** `{m.get('payload_source')}`")
        meta = m.get("meta") or {}
        lines.append(f"- **Date / venue / result:** {meta.get('date') or '—'} · {meta.get('venue') or '—'} · {meta.get('winner') or '—'} ({meta.get('margin') or '—'})")
        cu = m.get("conditions_used") or {}
        lines.append(
            f"- **Conditions:** venue={cu.get('venue_profile') or '—'} · "
            f"match_time={cu.get('match_time') or '—'} · toss=unknown (audit default)"
        )
        stc = m.get("squad_temporal_confound") or {}
        if isinstance(stc, dict) and stc.get("roster_drift_likely"):
            lines.append(
                f"- **Squad vs scorecard season:** match year **{stc.get('match_date_year', '—')}** vs "
                f"squad snapshot year **{stc.get('squad_snapshot_assumed_calendar_year', '—')}** — "
                f"_{stc.get('note', '')}_"
            )
        lines.append("")

        avp = m.get("actual_vs_predicted") or {}
        for team_label in sorted(avp.keys()):
            block = avp.get(team_label) or {}
            if not isinstance(block, dict):
                continue
            lines.append(f"#### {team_label}")
            lines.append("")
            lines.append(f"- **Predicted XI:** {_fmt_list(list(block.get('predicted_xi') or []))}")
            lines.append(f"- **Actual XI (scorecard):** {_fmt_list(list(block.get('actual_xi') or []))}")
            lines.append(
                f"- **Overlap count (registry-aware audit keys):** {block.get('correct_picks', 0)} / 11"
            )
            if int(block.get("registry_bridged_overlap_count") or 0) > 0:
                lines.append(
                    f"  - **Registry-bridged pairs** (same player, different scorecard vs squad spelling): "
                    f"{block.get('registry_bridged_overlap_count')}"
                )
            lines.append(f"- **Missed (actual not predicted):** {_fmt_list(list(block.get('missed_actual') or []))}")
            lines.append(f"- **Extra (predicted not actual):** {_fmt_list(list(block.get('extra_predicted') or []))}")
            bo = block.get("batting_order_overlap") or {}
            lines.append(
                f"- **Batting order:** top-3 positional matches={bo.get('top3_positional', '—')} · "
                f"openers(2)={bo.get('openers_positional', '—')} · "
                f"middle slots 4–7 (set overlap)={bo.get('middle_slots_4_7_set_overlap', '—')} · "
                f"lower 8–11 (set)={bo.get('lower_8_11_set', '—')}"
            )
            bu = block.get("bowling_usage") or {}
            lines.append(
                f"- **Bowling (scorecard vs predicted XI):** "
                f"distinct bowlers who bowled={bu.get('actual_distinct_bowlers_count', '—')} · "
                f"predicted XI classified as bowling options={bu.get('predicted_xi_bowling_options_count', '—')} · "
                f"bowlers who bowled and were in predicted XI={bu.get('actual_bowlers_also_in_predicted_xi_count', '—')}"
            )
            if bu.get("actual_bowlers_not_in_predicted_xi"):
                lines.append(
                    f"  - **Bowled but not in predicted XI:** {_fmt_list(list(bu.get('actual_bowlers_not_in_predicted_xi') or []))}"
                )
            if bu.get("predicted_xi_bowling_options_who_did_not_bowl"):
                lines.append(
                    f"  - **Predicted XI bowling options with no recorded overs:** {_fmt_list(list(bu.get('predicted_xi_bowling_options_who_did_not_bowl') or []))}"
                )
            imp = block.get("impact_subs") or {}
            lines.append(
                f"- **Impact subs (model):** {_fmt_list(list(imp.get('predicted_model_order') or []))}"
            )
            lines.append(f"  - **Actual (scorecard):** {imp.get('actual_from_scorecard') or 'not in parser schema'}")
            lines.append(f"  - _{imp.get('note', '')}_")
            tags = block.get("likely_gap_tags") or []
            if tags:
                lines.append(f"- **Mismatch notes (heuristic):** {', '.join(str(t) for t in tags)}")
            lines.append("")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append("PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python tools/generate_prediction_vs_actual_report.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="*", default=None, help="IPLT20 match URLs (default: same as validate script).")
    ap.add_argument(
        "--from-sqlite",
        type=int,
        metavar="N",
        default=0,
        help="Use last N rows from match_results.raw_payload (actual ingested data) instead of live fetch.",
    )
    ap.add_argument(
        "--audit-json",
        type=Path,
        default=None,
        help="Build report from a saved JSON file (output of validate_last_ipl_2026_matches.py --json).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "docs" / "prediction_vs_actual_report.md",
        help="Markdown output path.",
    )
    args = ap.parse_args()
    if args.audit_json:
        audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
        audit.setdefault("payload_source_note", f"Loaded audit JSON: {args.audit_json}")
    elif int(args.from_sqlite or 0) > 0:
        payloads = load_match_payloads_from_sqlite(int(args.from_sqlite))
        audit = run_audit(payloads_by_url=payloads)
        audit.setdefault(
            "payload_source_note",
            f"SQLite ``match_results`` preloaded payloads (last {len(payloads)} rows with team_a/team_b).",
        )
    else:
        urls = list(args.urls) if args.urls else list(DEFAULT_MATCH_URLS)
        audit = run_audit(urls)
        audit.setdefault("payload_source_note", "Live ``parse_scorecard`` for each URL (may fail if IPL HTML omits teams).")
    md = build_markdown(audit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out} ({len(md)} chars)")


if __name__ == "__main__":
    main()
