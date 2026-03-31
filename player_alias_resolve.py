"""
Resolve current-squad display names to SQLite / Cricsheet ``player_key`` values.

Squad **display names** are never altered here — only a lookup key for raw tables is produced.

Layers:
  A) Exact normalized key in franchise history.
  B) Deterministic Cricsheet-style variants (unsafe two-token ``x surname`` filtered unless unique in D).
  C) Same-surname keys: strict given-name alignment **or** initials-style history keys whose initials
     are prefix-compatible with squad tokens (e.g. ``ybk jaiswal`` ↔ multi-word squad).
  D) Franchise-only safe fallbacks:
     D1 — exactly one ``<initial> <surname>``-shaped key for that surname + initial.
     D2 — exactly one history key for that surname and strict given alignment.
     D3 — exactly one key for that surname, history given is initials-style, and initials
          are prefix-compatible with the squad (never for full-token given names like ``raghu``).

``ambiguous_alias`` means multiple candidates — never pick arbitrarily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, FrozenSet, Optional

import learner

_SURNAME_PARTICLES: frozenset[str] = frozenset(
    {
        "de",
        "van",
        "der",
        "den",
        "da",
        "di",
        "du",
        "von",
        "st",
        "ste",
        "le",
        "la",
        "el",
        "al",
        "ibn",
        "ben",
        "bin",
        "del",
        "della",
        "ter",
        "ten",
        "op",
        "mac",
        "mc",
    }
)


@dataclass(frozen=True)
class PlayerHistoryResolution:
    squad_full_name: str
    normalized_full_name_key: str
    resolved_history_key: Optional[str]
    resolution_type: str  # exact_match | alias_match | ambiguous_alias | no_match
    confidence: float
    ambiguous_candidates: tuple[str, ...]
    layer_b_variant_hits: tuple[str, ...]
    resolution_layer_used: str = "unresolved"
    surname_candidates_checked: tuple[str, ...] = ()
    layer_d_reason: str = ""
    layer_d_branch: str = ""
    surname_bucket_size: int = 0


def split_given_surname_tokens(norm_tokens: list[str]) -> tuple[list[str], list[str]]:
    if not norm_tokens:
        return [], []
    if len(norm_tokens) == 1:
        return [], norm_tokens[:]
    if len(norm_tokens) >= 3 and norm_tokens[-2] in _SURNAME_PARTICLES:
        return norm_tokens[:-2], norm_tokens[-2:]
    return norm_tokens[:-1], norm_tokens[-1:]


def _initial_sig_from_given_tokens(given: list[str]) -> str:
    parts: list[str] = []
    for g in given:
        g = (g or "").strip()
        if not g:
            continue
        if len(g) == 1:
            parts.append(g)
        elif len(g) == 2 and g.isalpha():
            parts.append(g[0])
            parts.append(g[1])
        else:
            parts.append(g[0])
    return "".join(parts)


def _variant_strings(given: list[str], surname_tokens: list[str]) -> list[str]:
    surname = " ".join(surname_tokens).strip()
    if not surname:
        return []
    out: list[str] = []
    if given:
        initials = "".join((g[0] for g in given if g))
        if initials:
            out.append(f"{initials} {surname}")
        out.append(f"{given[0][0]} {surname}")
        if len(given[0]) >= 2:
            out.append(f"{given[0][0]}{given[0][1]} {surname}")
        if len(given) >= 2:
            out.append(f"{given[0][0]}{given[1][0]} {surname}")
    return out


def _normalized_variants_from_squad_key(pk: str) -> list[str]:
    tokens = [t for t in (pk or "").split() if t]
    given, sur = split_given_surname_tokens(tokens)
    raw = _variant_strings(given, sur)
    seen: set[str] = set()
    out: list[str] = []
    for s in raw:
        nk = learner.normalize_player_key(s)
        if nk and nk not in seen:
            seen.add(nk)
            out.append(nk)
    return out


def _parse_history_key_tokens(key: str) -> tuple[list[str], list[str]]:
    return split_given_surname_tokens([t for t in (key or "").split() if t])


def _surname_tokens_match(history_key: str, squad_surname_tokens: list[str]) -> bool:
    if not squad_surname_tokens:
        return False
    hk = (history_key or "").strip()
    if not hk:
        return False
    squad_sur = " ".join(squad_surname_tokens).strip()
    if hk == squad_sur:
        return True
    if hk.endswith(" " + squad_sur):
        return True
    _g_hist, s_hist = _parse_history_key_tokens(hk)
    if not s_hist:
        return False
    return s_hist == squad_surname_tokens


def _unsafe_single_letter_plus_two_token_key(squad_given: list[str], history_key: str) -> bool:
    if not squad_given:
        return False
    hg, _hsur = _parse_history_key_tokens(history_key)
    if not hg:
        return False
    h0 = hg[0]
    s0 = squad_given[0]
    if len(h0) != 1 or len(s0) <= 1:
        return False
    if s0[0] != h0:
        return False
    ntok = len(history_key.split())
    if ntok >= 3:
        return False
    return True


def _strict_given_alignment(squad_given: list[str], history_key: str) -> bool:
    hg, _ = _parse_history_key_tokens(history_key)
    if not squad_given and not hg:
        return True
    if not squad_given or not hg:
        return False
    s0 = squad_given[0]
    h0 = hg[0]
    if s0 == h0:
        return True
    if min(len(s0), len(h0)) >= 3 and (s0.startswith(h0) or h0.startswith(s0)):
        return True
    ssig = _initial_sig_from_given_tokens(squad_given)
    hsig = _initial_sig_from_given_tokens(hg)
    if ssig and hsig:
        if ssig == hsig:
            return s0 == h0
        if len(ssig) == 1 and len(hsig) >= 2 and hsig.startswith(ssig):
            return True
        if len(hsig) == 1 and len(ssig) >= 2 and ssig.startswith(hsig):
            return True
    return False


def _history_given_is_initials_style(hg: list[str]) -> bool:
    """True for ``r`` ``l`` or ``rl`` / ``ybk`` / ``ra``; false for full tokens like ``raghu``."""
    if not hg:
        return False
    if all(len(t) == 1 for t in hg):
        return True
    if len(hg) == 1:
        t = hg[0]
        if 2 <= len(t) <= 4 and t.isalpha():
            return True
    return False


def _history_initials_blob(hg: list[str]) -> str:
    if not hg:
        return ""
    if all(len(tok) == 1 for tok in hg):
        return "".join(hg).lower()
    if len(hg) == 1 and 2 <= len(hg[0]) <= 4 and hg[0].isalpha():
        return hg[0].lower()
    return ""


def _single_word_hist_blob_subsequence(W: str, hb: str) -> bool:
    """Match ``ra`` to ``ravindra``, reject ``rl`` to ``rohit`` (no ``l`` after ``r``)."""
    if not W or not hb:
        return False
    W = W.lower()
    hb = hb.lower()
    if W[0] != hb[0]:
        return False
    if W.startswith(hb):
        return True
    pos = 1
    for i in range(1, len(hb)):
        found = W.find(hb[i], pos)
        if found < 0:
            return False
        pos = found + 1
    return True


def _initials_prefix_compatible(
    squad_given: list[str],
    history_key: str,
    *,
    surname_bucket_size: int = 999,
) -> bool:
    """
    History key given part is initials-style; initials align with squad tokens (prefix rules).
    Multi-word: ``hb`` is a prefix of per-token first letters. Single-word: ``W.startswith(hb)``
    or ordered subsequence for extra letters (``rl`` / ``ra`` in ``rahul`` / ``ravindra``).

    When ``surname_bucket_size == 1`` and the history blob has **extra** initials (e.g. ``ybk`` for
    display ``Yashasvi`` only), accept if those extra letters do not appear in the first name
    (omitted middle names in Cricsheet). Still blocks two-letter keys like ``rj`` → ``rohit``.
    """
    hg, _ = _parse_history_key_tokens(history_key)
    if not squad_given or not _history_given_is_initials_style(hg):
        return False
    hb = _history_initials_blob(hg)
    if not hb:
        return False
    words = [g.lower() for g in squad_given if g]
    if not words:
        return False
    sb_join = "".join(w[0] for w in words)
    if not sb_join:
        return False
    if hb == sb_join:
        return True
    if len(words) >= 2:
        if sb_join.startswith(hb):
            return True
        if len(hb) <= len(sb_join) and sb_join[: len(hb)] == hb:
            return True
        if (
            surname_bucket_size == 1
            and len(hb) > len(sb_join)
            and hb.startswith(sb_join)
            and len(hb) >= 3
        ):
            W = words[0]
            suffix = hb[len(sb_join) :]
            if suffix and all(ch not in W for ch in suffix):
                return True
        return False
    W = words[0]
    if W.startswith(hb):
        return True
    if sb_join.startswith(hb):
        return True
    if len(hb) > len(sb_join) and hb.startswith(sb_join):
        if (
            surname_bucket_size == 1
            and len(words) == 1
            and len(hb) >= 3
        ):
            suffix = hb[len(sb_join) :]
            if suffix and all(ch not in W for ch in suffix):
                return True
        return _single_word_hist_blob_subsequence(W, hb)
    return False


def _layer_b_candidates_filtered(
    squad_given: list[str],
    squad_surname_tokens: list[str],
    variants: list[str],
    franchise_keys: FrozenSet[str],
) -> set[str]:
    raw = {v for v in variants if v in franchise_keys}
    safe: set[str] = set()
    for k in raw:
        if _unsafe_single_letter_plus_two_token_key(squad_given, k):
            continue
        _hg, hsur = _parse_history_key_tokens(k)
        if tuple(hsur) != tuple(squad_surname_tokens):
            continue
        safe.add(k)
    return safe


def _layer_c_candidates(
    squad_given: list[str],
    squad_surname_tokens: list[str],
    franchise_keys: FrozenSet[str],
    surname_bucket_size: int,
) -> set[str]:
    """Strict given alignment OR initials-style history key prefix-compatible with squad."""
    out: set[str] = set()
    if not squad_given:
        return out
    for k in franchise_keys:
        if not _surname_tokens_match(k, squad_surname_tokens):
            continue
        if _strict_given_alignment(squad_given, k) or _initials_prefix_compatible(
            squad_given, k, surname_bucket_size=surname_bucket_size
        ):
            out.add(k)
    return out


def _all_keys_for_surname(
    squad_surname_tokens: list[str],
    franchise_keys: FrozenSet[str],
) -> tuple[str, ...]:
    if not squad_surname_tokens:
        return ()
    found = [k for k in franchise_keys if _surname_tokens_match(k, squad_surname_tokens)]
    return tuple(sorted(found))


def _layer_d_single_initial_unique(
    squad_given: list[str],
    squad_surname_tokens: list[str],
    franchise_keys: FrozenSet[str],
) -> tuple[Optional[str], tuple[str, ...], str]:
    """
    Exactly one franchise key shaped like ``<one-letter> <surname…>`` with matching initial.
    Franchise-scoped; multiple such keys → no match (ambiguous).
    """
    if not squad_given or not squad_surname_tokens:
        return None, (), "layer_d1_skip_no_given_or_surname"
    s0 = squad_given[0]
    if not s0:
        return None, (), "layer_d1_skip_empty_given"
    f0 = s0[0]
    candidates: list[str] = []
    for k in franchise_keys:
        if not _surname_tokens_match(k, squad_surname_tokens):
            continue
        hg, hsur = _parse_history_key_tokens(k)
        if tuple(hsur) != tuple(squad_surname_tokens):
            continue
        if not hg:
            continue
        h0 = hg[0]
        if len(h0) != 1:
            continue
        if h0 != f0:
            continue
        candidates.append(k)
    checked = tuple(sorted(candidates))
    if len(candidates) == 1:
        return candidates[0], checked, "layer_d1_accepted_unique_single_initial"
    if not candidates:
        return None, checked, "layer_d1_no_single_initial_candidates"
    return None, checked, f"layer_d1_reject_multiple_n={len(candidates)}"


def _layer_d_unique_surname_strict(
    squad_given: list[str],
    squad_surname_tokens: list[str],
    franchise_keys: FrozenSet[str],
) -> tuple[Optional[str], tuple[str, ...], str]:
    """
    Exactly one raw key for this surname in the franchise, and given names are not contradictory.
    Blocks Rohit → sole ``raghu sharma`` (strict_given_alignment fails).
    """
    if not squad_surname_tokens:
        return None, (), "layer_d2_skip_no_surname"
    bucket = [k for k in franchise_keys if _surname_tokens_match(k, squad_surname_tokens)]
    checked = tuple(sorted(bucket))
    if len(bucket) != 1:
        return None, checked, f"layer_d2_reject_surname_bucket_size={len(bucket)}"
    only = bucket[0]
    if not squad_given:
        return only, checked, "layer_d2_accepted_mononym_bucket"
    if _strict_given_alignment(squad_given, only):
        return only, checked, "layer_d2_accepted_unique_surname_strict_ok"
    return None, checked, "layer_d2_reject_unique_surname_given_contradicts"


def _layer_d_relaxed_unique_surname(
    squad_given: list[str],
    squad_surname_tokens: list[str],
    franchise_keys: FrozenSet[str],
) -> tuple[Optional[str], tuple[str, ...], str]:
    """
    Single surname bucket, history given is initials-only style, prefix-compatible with squad.
    Blocks ``raghu sharma`` for ``Rohit Sharma`` (full given token → not initials style).
    """
    if not squad_surname_tokens:
        return None, (), "layer_d3_skip_no_surname"
    bucket = [k for k in franchise_keys if _surname_tokens_match(k, squad_surname_tokens)]
    checked = tuple(sorted(bucket))
    if len(bucket) != 1:
        return None, checked, f"layer_d3_reject_surname_bucket_size={len(bucket)}"
    only = bucket[0]
    hg, _ = _parse_history_key_tokens(only)
    if not _history_given_is_initials_style(hg):
        return None, checked, "layer_d3_reject_history_not_initials_style"
    if not squad_given:
        return None, checked, "layer_d3_reject_no_squad_given"
    if not _initials_prefix_compatible(squad_given, only, surname_bucket_size=1):
        return None, checked, "layer_d3_reject_prefix_incompatible"
    return only, checked, "layer_d3_accepted_relaxed_unique_surname"


def _finalize(
    *,
    name: str,
    pk: str,
    resolved: Optional[str],
    rtype: str,
    confidence: float,
    ambiguous: tuple[str, ...],
    b_raw: tuple[str, ...],
    layer: str,
    surname_checked: tuple[str, ...],
    d_reason: str,
    d_branch: str,
    surname_bucket_size: int,
) -> PlayerHistoryResolution:
    return PlayerHistoryResolution(
        squad_full_name=name,
        normalized_full_name_key=pk,
        resolved_history_key=resolved,
        resolution_type=rtype,
        confidence=confidence,
        ambiguous_candidates=ambiguous,
        layer_b_variant_hits=b_raw,
        resolution_layer_used=layer,
        surname_candidates_checked=surname_checked,
        layer_d_reason=d_reason,
        layer_d_branch=d_branch,
        surname_bucket_size=surname_bucket_size,
    )


def resolve_player_to_history_key(
    squad_full_name: str,
    franchise_player_keys: FrozenSet[str],
) -> PlayerHistoryResolution:
    name = (squad_full_name or "").strip()
    pk = learner.normalize_player_key(name)
    if not name or not pk:
        return _finalize(
            name=name,
            pk=pk,
            resolved=None,
            rtype="no_match",
            confidence=0.0,
            ambiguous=(),
            b_raw=(),
            layer="unresolved",
            surname_checked=(),
            d_reason="empty_name",
            d_branch="",
            surname_bucket_size=0,
        )

    tokens = [t for t in pk.split() if t]
    squad_given, squad_sur = split_given_surname_tokens(tokens)
    surname_all = _all_keys_for_surname(squad_sur, franchise_player_keys)
    bucket_n = len(surname_all)

    if pk in franchise_player_keys:
        return _finalize(
            name=name,
            pk=pk,
            resolved=pk,
            rtype="exact_match",
            confidence=1.0,
            ambiguous=(),
            b_raw=(),
            layer="exact",
            surname_checked=(),
            d_reason="",
            d_branch="",
            surname_bucket_size=bucket_n,
        )

    variants = _normalized_variants_from_squad_key(pk)
    b_raw = tuple(sorted({v for v in variants if v in franchise_player_keys}))
    b_hits = tuple(sorted(_layer_b_candidates_filtered(squad_given, squad_sur, variants, franchise_player_keys)))

    if len(b_hits) == 1:
        return _finalize(
            name=name,
            pk=pk,
            resolved=b_hits[0],
            rtype="alias_match",
            confidence=0.92,
            ambiguous=(),
            b_raw=b_raw,
            layer="layer_b",
            surname_checked=(),
            d_reason="",
            d_branch="",
            surname_bucket_size=bucket_n,
        )
    if len(b_hits) > 1:
        return _finalize(
            name=name,
            pk=pk,
            resolved=None,
            rtype="ambiguous_alias",
            confidence=0.0,
            ambiguous=b_hits,
            b_raw=b_raw,
            layer="ambiguous",
            surname_checked=(),
            d_reason="layer_b_multiple_variant_hits",
            d_branch="",
            surname_bucket_size=bucket_n,
        )

    c_hits = _layer_c_candidates(squad_given, squad_sur, franchise_player_keys, bucket_n)
    c_hits.discard(pk)
    if len(c_hits) == 1:
        only = next(iter(c_hits))
        return _finalize(
            name=name,
            pk=pk,
            resolved=only,
            rtype="alias_match",
            confidence=0.78,
            ambiguous=(),
            b_raw=b_raw,
            layer="layer_c",
            surname_checked=tuple(sorted(c_hits)),
            d_reason="",
            d_branch="",
            surname_bucket_size=bucket_n,
        )
    if len(c_hits) > 1:
        amb = tuple(sorted(c_hits))
        return _finalize(
            name=name,
            pk=pk,
            resolved=None,
            rtype="ambiguous_alias",
            confidence=0.0,
            ambiguous=amb,
            b_raw=b_raw,
            layer="ambiguous",
            surname_checked=amb,
            d_reason="layer_c_multiple_candidates",
            d_branch="",
            surname_bucket_size=bucket_n,
        )

    d1_key, d1_checked, d1_reason = _layer_d_single_initial_unique(
        squad_given, squad_sur, franchise_player_keys
    )
    if d1_key:
        return _finalize(
            name=name,
            pk=pk,
            resolved=d1_key,
            rtype="alias_match",
            confidence=0.72,
            ambiguous=(),
            b_raw=b_raw,
            layer="layer_d",
            surname_checked=d1_checked,
            d_reason=d1_reason,
            d_branch="single_initial_unique",
            surname_bucket_size=bucket_n,
        )

    d2_key, d2_checked, d2_reason = _layer_d_unique_surname_strict(
        squad_given, squad_sur, franchise_player_keys
    )
    if d2_key:
        return _finalize(
            name=name,
            pk=pk,
            resolved=d2_key,
            rtype="alias_match",
            confidence=0.68,
            ambiguous=(),
            b_raw=b_raw,
            layer="layer_d",
            surname_checked=d2_checked,
            d_reason=d2_reason,
            d_branch="unique_surname_strict",
            surname_bucket_size=bucket_n,
        )

    d3_key, d3_checked, d3_reason = _layer_d_relaxed_unique_surname(
        squad_given, squad_sur, franchise_player_keys
    )
    if d3_key:
        return _finalize(
            name=name,
            pk=pk,
            resolved=d3_key,
            rtype="alias_match",
            confidence=0.65,
            ambiguous=(),
            b_raw=b_raw,
            layer="layer_d_relaxed_unique_surname",
            surname_checked=d3_checked,
            d_reason=d3_reason,
            d_branch="relaxed_unique_surname",
            surname_bucket_size=bucket_n,
        )

    layer_d_combined_reason = f"d1:{d1_reason};d2:{d2_reason};d3:{d3_reason}"
    return _finalize(
        name=name,
        pk=pk,
        resolved=None,
        rtype="no_match",
        confidence=0.0,
        ambiguous=(),
        b_raw=b_raw,
        layer="unresolved",
        surname_checked=surname_all if surname_all else d1_checked,
        d_reason=layer_d_combined_reason,
        d_branch="rejected",
        surname_bucket_size=bucket_n,
    )


def history_lookup_key_from_resolution(res: PlayerHistoryResolution) -> Optional[str]:
    if res.resolution_type in ("exact_match", "alias_match"):
        return res.resolved_history_key
    return None


def ambiguous_candidates_json(res: PlayerHistoryResolution) -> Optional[str]:
    if not res.ambiguous_candidates:
        return None
    return json.dumps(list(res.ambiguous_candidates))


def history_status_from_resolution(res: PlayerHistoryResolution) -> str:
    if res.resolution_type == "exact_match":
        return "exact_linked"
    if res.resolution_type == "alias_match":
        return "alias_linked"
    if res.resolution_type == "ambiguous_alias":
        return "ambiguous_alias"
    if res.resolution_type == "ambiguous_alias_collision":
        return "history_key_collision_loser"
    return "no_history_match"


def squad_history_alignment_score(
    squad_display_name: str,
    history_lookup_key: str,
    matched_sample_name: Optional[str],
) -> float:
    """
    0–1 score: how well the squad display name aligns with the resolved SQLite key / sample name.
    Used for deterministic collision tie-breaks between two squad players sharing one key.
    """
    pk = learner.normalize_player_key(str(squad_display_name or "").strip())
    if not pk or not (history_lookup_key or "").strip():
        return 0.0
    tokens = [t for t in pk.split() if t]
    given, _sur = split_given_surname_tokens(tokens)
    hk = str(history_lookup_key).strip()
    if _strict_given_alignment(given, hk):
        return 1.0
    if matched_sample_name and str(matched_sample_name).strip():
        mk = learner.normalize_player_key(str(matched_sample_name).strip())
        if mk and _strict_given_alignment(given, mk):
            return 0.9
    if pk == hk:
        return 0.95
    if hk and (pk in hk or hk in pk):
        return 0.55
    return 0.25


def squad_given_surname_alignment_scores(
    squad_display_name: str,
    history_lookup_key: str,
    matched_sample_name: Optional[str],
) -> tuple[float, float]:
    """
    Separate 0–1 given-name and surname alignment vs resolved key / sample (collision tie-break).
    """
    pk = learner.normalize_player_key(str(squad_display_name or "").strip())
    if not pk:
        return 0.0, 0.0
    tokens = [t for t in pk.split() if t]
    given, sur = split_given_surname_tokens(tokens)
    hk = str(history_lookup_key or "").strip()
    g_score = 0.0
    if hk and _strict_given_alignment(given, hk):
        g_score = 1.0
    elif matched_sample_name and str(matched_sample_name).strip():
        mk = learner.normalize_player_key(str(matched_sample_name).strip())
        if mk and _strict_given_alignment(given, mk):
            g_score = 0.92
    if g_score == 0.0 and hk and given:
        if _initials_prefix_compatible(given, hk):
            g_score = 0.55

    s_score = 0.0
    if hk and _surname_tokens_match(hk, sur):
        s_score = 1.0
    elif matched_sample_name and str(matched_sample_name).strip():
        mk_n = learner.normalize_player_key(str(matched_sample_name).strip())
        if mk_n and _surname_tokens_match(mk_n, sur):
            s_score = 0.95
    if s_score == 0.0 and sur and hk:
        sur_s = " ".join(sur).strip()
        if sur_s and (sur_s in hk or hk.endswith(" " + sur_s)):
            s_score = 0.5

    return g_score, s_score


def rolled_up_history_interpretation(
    res: PlayerHistoryResolution,
    *,
    distinct_franchise_matches: int,
    franchise_xi_row_count: int,
    franchise_key_count: int,
    role_bucket: Optional[str],
    usable_history_rows: int = 0,
) -> str:
    """
    Rollup: linked_ok | linked_low_sample | likely_alias_miss | likely_new_or_sparse.
    """
    import config

    low_max = int(getattr(config, "STAGE_F_LINKED_LOW_SAMPLE_MAX_ROWS", 3))

    if res.resolution_type in ("exact_match", "alias_match"):
        if usable_history_rows <= 0:
            return "likely_new_or_sparse"
        if usable_history_rows > low_max:
            return "linked_ok"
        return "linked_low_sample"

    depth_ok = distinct_franchise_matches >= int(
        getattr(config, "STAGE_F_FRANCHISE_DEPTH_OK_MATCHES", 10)
    )
    index_ok = franchise_key_count >= int(getattr(config, "STAGE_F_FRANCHISE_KEY_INDEX_OK", 25))
    xi_ok = franchise_xi_row_count >= int(getattr(config, "STAGE_F_FRANCHISE_XI_ROWS_OK", 40))

    sparse_franchise = not (depth_ok and (index_ok or xi_ok))

    core_roles = frozenset(
        {
            "Batter",
            "WK-Batter",
            "All-Rounder",
        }
    )
    core = (role_bucket or "") in core_roles

    if res.resolution_type == "no_match":
        if res.surname_bucket_size > 0 or len(res.surname_candidates_checked) > 0:
            return "likely_alias_miss"
        return "likely_new_or_sparse"

    if res.resolution_type == "ambiguous_alias":
        if sparse_franchise or not core:
            return "likely_new_or_sparse"
        if depth_ok:
            return "likely_alias_miss"
        return "likely_new_or_sparse"

    if res.resolution_type == "ambiguous_alias_collision":
        return "history_key_collision_loser"

    if sparse_franchise or not core:
        return "likely_new_or_sparse"

    return "likely_new_or_sparse"


@dataclass(frozen=True)
class DebutantAliasSuppressionOutcome:
    """After optional debutant guard: effective keys and resolutions for Stage F / history_xi."""

    effective_history_lookup_key: Optional[str]
    effective_global_resolved_key: Optional[str]
    franchise_resolution_effective: PlayerHistoryResolution
    global_resolution_effective: Optional[PlayerHistoryResolution]
    suppression_applied: bool
    debutant_alias_rejection_reason: str
    likely_first_ipl_player: bool


def _synthetic_no_match_resolution(base: PlayerHistoryResolution) -> PlayerHistoryResolution:
    """Treat weak alias as unlink for rollup/debug (display only)."""
    return PlayerHistoryResolution(
        squad_full_name=base.squad_full_name,
        normalized_full_name_key=base.normalized_full_name_key,
        resolved_history_key=None,
        resolution_type="no_match",
        confidence=0.0,
        ambiguous_candidates=(),
        layer_b_variant_hits=base.layer_b_variant_hits,
        resolution_layer_used="debutant_suppression",
        surname_candidates_checked=base.surname_candidates_checked,
        layer_d_reason="weak_alias_suppressed_as_debutant_risk",
        layer_d_branch="debutant_guard",
        surname_bucket_size=base.surname_bucket_size,
    )


def apply_debutant_alias_suppression(
    *,
    franchise_res: PlayerHistoryResolution,
    global_res: Optional[PlayerHistoryResolution],
    history_lookup_key: Optional[str],
    global_resolved_key: Optional[str],
    franchise_history_row_count: int,
    global_distinct_for_franchise_key: int,
    global_distinct_for_global_key: int,
) -> DebutantAliasSuppressionOutcome:
    """
    Suppress unsafe alias links for likely first-IPL players when evidence is only weak/medium
    and franchise + global sample depth for the resolved key are thin.
    """
    import config

    eff_ek = history_lookup_key
    eff_grk = global_resolved_key
    disp_fr = franchise_res
    disp_glob = global_res
    reasons: list[str] = []
    suppressed = False

    strong_conf = float(getattr(config, "DEBUTANT_ALIAS_CONFIDENCE_STRONG", 0.915))
    min_g = int(getattr(config, "DEBUTANT_MIN_GLOBAL_DISTINCT_FOR_WEAK_ALIAS", 9))
    min_fr = int(getattr(config, "DEBUTANT_MIN_FRANCHISE_HISTORY_ROWS_FOR_WEAK_ALIAS", 4))

    if franchise_res.resolution_type == "alias_match" and history_lookup_key:
        c = float(franchise_res.confidence)
        strong_global = global_distinct_for_franchise_key >= min_g
        strong_fr = franchise_history_row_count >= min_fr
        weak_alias = c < strong_conf
        if weak_alias and not (strong_global or strong_fr):
            eff_ek = None
            eff_grk = None
            disp_fr = _synthetic_no_match_resolution(franchise_res)
            disp_glob = None
            suppressed = True
            reasons.append(
                "franchise_alias_below_strong_confidence_and_thin_franchise_rows_and_global_distinct"
            )

    if (
        not suppressed
        and franchise_res.resolution_type == "no_match"
        and global_res is not None
        and global_res.resolution_type == "alias_match"
        and global_resolved_key
    ):
        c = float(global_res.confidence)
        strong_global = global_distinct_for_global_key >= min_g
        weak_alias = c < strong_conf
        if weak_alias and not strong_global:
            eff_grk = None
            disp_glob = None
            suppressed = True
            reasons.append("global_alias_below_strong_confidence_and_thin_global_distinct")

    reason_str = "; ".join(reasons) if reasons else ""
    likely_debut = suppressed or (
        eff_ek is None
        and eff_grk is None
        and franchise_res.resolution_type != "exact_match"
    )

    return DebutantAliasSuppressionOutcome(
        effective_history_lookup_key=eff_ek,
        effective_global_resolved_key=eff_grk,
        franchise_resolution_effective=disp_fr,
        global_resolution_effective=disp_glob,
        suppression_applied=suppressed,
        debutant_alias_rejection_reason=reason_str,
        likely_first_ipl_player=bool(likely_debut),
    )


def rolled_up_with_global_alias_fallback(
    franchise_res: PlayerHistoryResolution,
    franchise_rolled: str,
    global_res: Optional[PlayerHistoryResolution],
    global_distinct_matches: int,
) -> str:
    """
    When franchise linkage is ``no_match`` but an IPL-wide alias pass finds a stored key with rows,
    avoid reporting ``likely_new_or_sparse`` for that name.
    """
    if franchise_res.resolution_type in ("exact_match", "alias_match"):
        return franchise_rolled
    gk = history_lookup_key_from_resolution(global_res) if global_res else None
    if not gk or global_distinct_matches <= 0:
        return franchise_rolled
    import config

    low_max = int(getattr(config, "STAGE_F_LINKED_LOW_SAMPLE_MAX_ROWS", 3))
    if global_distinct_matches > low_max:
        return "linked_ok_via_global_alias"
    return "linked_low_sample_via_global_alias"


def classify_stage_f_team_health(
    *,
    per_player_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """
    Classify squad linkage health: healthy | healthy_with_sparse_new_players | partial_linkage_issue | major_linkage_failure.
    """
    import config

    core_roles = frozenset({"Batter", "WK-Batter", "All-Rounder"})
    core_rows = [r for r in per_player_rows if (r.get("role_bucket") or "") in core_roles]
    n_core = len(core_rows)
    core_linked = sum(
        1
        for r in core_rows
        if (r.get("resolution_type") or "") in ("exact_match", "alias_match")
        and (r.get("collision_resolution_outcome") or "") != "lost_collision"
    )
    core_unres = n_core - core_linked
    frac_core_bad = (core_unres / float(n_core)) if n_core else 0.0

    unres_rows = [
        r
        for r in per_player_rows
        if not (r.get("history_lookup_key") or "").strip()
        and (r.get("rolled_up_interpretation") or "")
        not in ("linked_ok", "linked_low_sample")
    ]
    unres_non_core = sum(1 for r in unres_rows if (r.get("role_bucket") or "") not in core_roles)
    unres_total = len(unres_rows)
    frac_unres_non_core = (unres_non_core / float(unres_total)) if unres_total else 0.0

    major_frac = float(getattr(config, "STAGE_F_MAJOR_CORE_UNRESOLVED_FRAC", 0.6))
    major_min_linked = int(getattr(config, "STAGE_F_MAJOR_MIN_CORE_LINKED", 5))
    partial_frac = float(getattr(config, "STAGE_F_PARTIAL_CORE_UNRESOLVED_FRAC", 0.3))
    new_frac = float(getattr(config, "STAGE_F_UNRESOLVED_MOSTLY_NEW_FRAC", 0.65))

    mostly_new_unres = sum(1 for r in unres_rows if int(r.get("surname_bucket_size") or 0) < 1)
    mostly_new = unres_total >= 1 and (mostly_new_unres / float(unres_total)) >= new_frac

    is_major = (
        n_core >= 1
        and frac_core_bad > major_frac
        and core_linked < major_min_linked
    )

    if is_major:
        health = "major_linkage_failure"
    elif frac_core_bad < partial_frac or mostly_new:
        if mostly_new and unres_total >= 1:
            health = "healthy_with_sparse_new_players"
        elif frac_core_bad < partial_frac:
            health = "healthy"
        else:
            health = "healthy_with_sparse_new_players"
    elif frac_core_bad > partial_frac:
        health = "partial_linkage_issue"
    else:
        health = "healthy"

    detail = {
        "core_players": n_core,
        "core_linked": core_linked,
        "core_unresolved": core_unres,
        "fraction_core_unresolved": round(frac_core_bad, 4),
        "unresolved_total": unres_total,
        "unresolved_non_core_fraction": round(frac_unres_non_core, 4),
        "unresolved_mostly_no_surname_keys": mostly_new,
        "unresolved_without_surname_keys": mostly_new_unres,
    }
    return health, detail


__all__ = [
    "DebutantAliasSuppressionOutcome",
    "PlayerHistoryResolution",
    "ambiguous_candidates_json",
    "apply_debutant_alias_suppression",
    "classify_stage_f_team_health",
    "history_lookup_key_from_resolution",
    "history_status_from_resolution",
    "resolve_player_to_history_key",
    "squad_history_alignment_score",
    "squad_given_surname_alignment_scores",
    "rolled_up_history_interpretation",
    "rolled_up_with_global_alias_fallback",
    "split_given_surname_tokens",
]
