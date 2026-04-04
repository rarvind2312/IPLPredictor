from __future__ import annotations

import json
import unittest
from pathlib import Path

import config
import player_registry


class TestPlayerRegistry(unittest.TestCase):
    def test_registry_uses_squad_json_for_cameron_green(self) -> None:
        players = player_registry.registry_players()
        record = players.get("cameron green") or {}
        self.assertEqual(record.get("display_name"), "Cameron Green")
        self.assertEqual(record.get("secondary_role"), "batting_allrounder")
        self.assertEqual(record.get("allrounder_type"), "batting_allrounder")
        self.assertEqual(record.get("marquee_tier"), "tier_1")
        self.assertEqual((record.get("field_sources") or {}).get("secondary_role"), "squad_json")
        self.assertEqual(((record.get("metadata_source_summary") or {}).get("preferred_metadata_source")), "squad_json")

    def test_alias_override_maps_rohit_to_rg(self) -> None:
        _canon_to_aliases, alias_to_canon = player_registry.registry_alias_override_maps()
        self.assertEqual(alias_to_canon.get("rohit sharma"), "rg sharma")

    def test_registry_marquee_lookup_uses_master_registry(self) -> None:
        marquee = player_registry.registry_marquee_lookup_map()
        self.assertEqual((marquee.get("rohit sharma") or {}).get("marquee_tier"), "tier_1")

    def test_registry_metadata_lookup_supports_history_key(self) -> None:
        meta = player_registry.registry_metadata_lookup_map()
        record = meta.get("rg sharma") or {}
        self.assertEqual(record.get("display_name"), "Rohit Sharma")

    def test_registry_handles_variant_surya_key(self) -> None:
        meta = player_registry.registry_metadata_lookup_map()
        record = meta.get("surya kumar yadav") or {}
        self.assertEqual(record.get("display_name"), "Suryakumar Yadav")
        self.assertEqual((player_registry.registry_marquee_lookup_map().get("surya kumar yadav") or {}).get("marquee_tier"), "tier_1")

    def test_registry_enriches_nitish_history_key(self) -> None:
        players = player_registry.registry_players()
        record = players.get("nitish kumar reddy") or {}
        self.assertEqual(record.get("history_canonical_key"), "nithish kumar reddy")
        self.assertEqual(record.get("team"), "Sunrisers Hyderabad")
        self.assertEqual(record.get("allrounder_type"), "batting_allrounder")
        self.assertEqual((record.get("field_sources") or {}).get("primary_role"), "squad_json")

    def test_registry_enriches_rinku_and_axar(self) -> None:
        players = player_registry.registry_players()
        rinku = players.get("rinku singh") or {}
        self.assertEqual(rinku.get("history_canonical_key"), "rk singh")
        self.assertEqual(rinku.get("likely_batting_band"), "middle_order")
        self.assertEqual((players.get("axar patel") or {}).get("history_canonical_key"), "ar patel")

    def test_registry_enriches_cameron_from_recent_ipl_evidence(self) -> None:
        players = player_registry.registry_players()
        record = players.get("cameron green") or {}
        self.assertEqual(record.get("history_canonical_key"), "c green")
        self.assertEqual((record.get("field_sources") or {}).get("history_canonical_key"), "db_linkage_enrichment")

    def test_registry_uses_raw_cricsheet_fallback_for_vaibhav(self) -> None:
        players = player_registry.registry_players()
        record = players.get("vaibhav sooryavanshi") or {}
        self.assertEqual(record.get("history_canonical_key"), "v suryavanshi")
        self.assertEqual(record.get("likely_batting_band"), "opener")
        self.assertEqual((record.get("field_sources") or {}).get("history_canonical_key"), "raw_cricsheet_fallback")

    def test_registry_uses_squad_json_for_abdul_samad(self) -> None:
        players = player_registry.registry_players()
        record = players.get("abdul samad") or {}
        self.assertEqual(record.get("primary_role"), "batter")
        self.assertEqual(record.get("role_description"), "batter")
        self.assertEqual(record.get("team"), "Lucknow Super Giants")
        self.assertEqual(((record.get("metadata_source_summary") or {}).get("preferred_metadata_source")), "squad_json")

    def test_registry_uses_squad_json_for_bumrah_and_abhishek(self) -> None:
        players = player_registry.registry_players()
        bumrah = players.get("jasprit bumrah") or {}
        self.assertEqual(bumrah.get("primary_role"), "bowler")
        self.assertEqual(bumrah.get("marquee_tier"), "tier_1")
        self.assertEqual((bumrah.get("field_sources") or {}).get("primary_role"), "squad_json")
        abhishek = players.get("abhishek sharma") or {}
        self.assertEqual(abhishek.get("allrounder_type"), "batting_allrounder")
        self.assertEqual((abhishek.get("field_sources") or {}).get("allrounder_type"), "squad_json")

    def test_registry_populates_batting_slot_constraints_from_metadata_defaults(self) -> None:
        players = player_registry.registry_players()
        rinku = players.get("rinku singh") or {}
        self.assertEqual(rinku.get("allowed_batting_slots"), [5, 6, 7])
        self.assertEqual(rinku.get("preferred_batting_slots"), [5, 6])
        self.assertFalse(rinku.get("opener_eligible"))
        bumrah = players.get("jasprit bumrah") or {}
        self.assertEqual(bumrah.get("allowed_batting_slots"), [9, 10, 11])
        self.assertEqual(bumrah.get("preferred_batting_slots"), [9, 10])
        self.assertFalse(bumrah.get("finisher_eligible"))

    def test_registry_preserves_existing_dubey_resolution(self) -> None:
        players = player_registry.registry_players()
        self.assertEqual((players.get("shubham dubey") or {}).get("history_canonical_key"), "sb dubey")
        self.assertEqual((players.get("saurabh dubey") or {}).get("history_canonical_key"), "")

    def test_registry_alias_map_includes_enriched_history_keys(self) -> None:
        _canon_to_aliases, alias_to_canon = player_registry.registry_alias_override_maps()
        self.assertEqual(alias_to_canon.get("nitish kumar reddy"), "nithish kumar reddy")

    def test_linkage_audit_artifact_exists(self) -> None:
        audit_path = Path(config.PLAYER_REGISTRY_LINKAGE_AUDIT_PATH)
        self.assertTrue(audit_path.is_file())
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        nitish = ((payload.get("players") or {}).get("nitish kumar reddy") or {})
        self.assertEqual(nitish.get("new_history_canonical_key"), "nithish kumar reddy")


if __name__ == "__main__":
    unittest.main()
