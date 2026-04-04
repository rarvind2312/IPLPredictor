"""Player alias resolver: squad identity vs SQLite ``player_key`` (strict, unambiguous)."""

from __future__ import annotations

import unittest

import player_alias_resolve


class TestPlayerAliasResolve(unittest.TestCase):
    def test_exact_match(self) -> None:
        keys = frozenset({"ruturaj gaikwad", "ms dhoni"})
        r = player_alias_resolve.resolve_player_to_history_key("Ruturaj Gaikwad", keys)
        self.assertEqual(r.resolution_type, "exact_match")
        self.assertEqual(r.resolved_history_key, "ruturaj gaikwad")
        self.assertEqual(r.confidence, 1.0)
        self.assertEqual(
            player_alias_resolve.history_status_from_resolution(r),
            "exact_linked",
        )

    def test_tilak_varma_exact(self) -> None:
        keys = frozenset({"tilak varma", "other"})
        r = player_alias_resolve.resolve_player_to_history_key("Tilak Varma", keys)
        self.assertEqual(r.resolution_type, "exact_match")
        self.assertEqual(r.resolved_history_key, "tilak varma")

    def test_layer_b_ambiguous_when_two_variants_exist(self) -> None:
        keys = frozenset({"rg sharma", "ro sharma"})
        r = player_alias_resolve.resolve_player_to_history_key("Rohit Gurunath Sharma", keys)
        self.assertEqual(r.resolution_type, "ambiguous_alias")
        self.assertIsNone(r.resolved_history_key)
        self.assertGreaterEqual(len(r.ambiguous_candidates), 2)

    def test_rohit_sharma_not_raghu_sharma(self) -> None:
        """Shared surname + same first initial must not map to a different given name."""
        keys = frozenset({"raghu sharma"})
        r = player_alias_resolve.resolve_player_to_history_key("Rohit Sharma", keys)
        self.assertNotEqual(r.resolved_history_key, "raghu sharma")
        self.assertEqual(r.resolution_type, "no_match")

    def test_rohit_sharma_can_match_rg_shorthand(self) -> None:
        keys = frozenset({"rg sharma"})
        r = player_alias_resolve.resolve_player_to_history_key("Rohit Sharma", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "rg sharma")

    def test_layer_d_r_sharma_when_unique_single_initial_in_franchise(self) -> None:
        """Layer D1: one ``<initial> <surname>`` key for that surname + initial → safe link."""
        keys = frozenset({"r sharma"})
        r = player_alias_resolve.resolve_player_to_history_key("Rohit Sharma", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "r sharma")
        self.assertIn(r.resolution_layer_used, ("layer_b", "layer_c", "layer_d"))

    def test_layer_c_ambiguous_two_initial_surname_expansions(self) -> None:
        """Two ``r <x> sharma`` keys both align with Rohit under Layer C → ambiguous."""
        keys = frozenset({"r g sharma", "r j sharma"})
        r = player_alias_resolve.resolve_player_to_history_key("Rohit Sharma", keys)
        self.assertEqual(r.resolution_type, "ambiguous_alias")
        self.assertIsNone(r.resolved_history_key)
        self.assertEqual(r.resolution_layer_used, "ambiguous")

    def test_layer_d_virat_v_kohli(self) -> None:
        keys = frozenset({"v kohli", "ms dhoni"})
        r = player_alias_resolve.resolve_player_to_history_key("Virat Kohli", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "v kohli")
        self.assertIn(r.resolution_layer_used, ("layer_b", "layer_c", "layer_d", "curated_alias_override"))

    def test_layer_b_single_hit_quinton_de_kock(self) -> None:
        keys = frozenset({"q de kock", "other player"})
        r = player_alias_resolve.resolve_player_to_history_key("Quinton de Kock", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "q de kock")

    def test_layer_c_hh_pandya(self) -> None:
        keys = frozenset({"hh pandya"})
        r = player_alias_resolve.resolve_player_to_history_key("Hardik Pandya", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "hh pandya")

    def test_no_match_empty_franchise(self) -> None:
        r = player_alias_resolve.resolve_player_to_history_key("Jane Doe", frozenset())
        self.assertEqual(r.resolution_type, "no_match")
        self.assertIsNone(r.resolved_history_key)

    def test_history_lookup_helper(self) -> None:
        keys = frozenset({"a b"})
        r = player_alias_resolve.resolve_player_to_history_key("A B", keys)
        hk = player_alias_resolve.history_lookup_key_from_resolution(r)
        self.assertEqual(hk, "a b")

    def test_layer_c_yashasvi_ybk_jaiswal(self) -> None:
        keys = frozenset({"ybk jaiswal", "other"})
        r = player_alias_resolve.resolve_player_to_history_key(
            "Yashasvi Bhupendra Kumar Jaiswal", keys
        )
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "ybk jaiswal")
        self.assertIn(r.resolution_layer_used, ("layer_b", "layer_c"))

    def test_yashasvi_jaiswal_short_display_ybk_unique_bucket(self) -> None:
        """Squad omits middle names; Cricsheet ``ybk`` with surname bucket 1."""
        keys = frozenset({"ybk jaiswal"})
        r = player_alias_resolve.resolve_player_to_history_key("Yashasvi Jaiswal", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "ybk jaiswal")

    def test_layer_d3_relaxed_unique_surname_rl_chahar(self) -> None:
        keys = frozenset({"rl chahar"})
        r = player_alias_resolve.resolve_player_to_history_key("Rahul Chahar", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "rl chahar")
        self.assertIn(
            r.resolution_layer_used,
            ("curated_alias_override", "layer_c", "layer_d_relaxed_unique_surname"),
        )

    def test_layer_d3_relaxed_direct_helper(self) -> None:
        """``_layer_d_relaxed_unique_surname`` when earlier layers would not apply."""
        keys = frozenset({"rl chahar"})
        k, _checked, reason = player_alias_resolve._layer_d_relaxed_unique_surname(
            ["rahul"],
            ["chahar"],
            keys,
        )
        self.assertEqual(k, "rl chahar")
        self.assertIn("accepted", reason)

    def test_layer_c_ra_jadeja(self) -> None:
        keys = frozenset({"ra jadeja"})
        r = player_alias_resolve.resolve_player_to_history_key("Ravindra Jadeja", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "ra jadeja")

    def test_layer_c_sv_samson(self) -> None:
        keys = frozenset({"sv samson"})
        r = player_alias_resolve.resolve_player_to_history_key("Sanju Samson", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "sv samson")

    def test_heinrich_h_klaasen_layer_d1(self) -> None:
        keys = frozenset({"h klaasen", "other"})
        r = player_alias_resolve.resolve_player_to_history_key("Heinrich Klaasen", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "h klaasen")
        self.assertIn(r.resolution_layer_used, ("layer_b", "layer_c", "layer_d", "curated_alias_override"))

    def test_ambiguous_two_prefix_compatible_jaiswal_keys(self) -> None:
        keys = frozenset({"ybk jaiswal", "yb jaiswal"})
        r = player_alias_resolve.resolve_player_to_history_key(
            "Yashasvi Bhupendra Kumar Jaiswal", keys
        )
        self.assertEqual(r.resolution_type, "ambiguous_alias")

    def test_curated_alias_override_rachin_ravindra(self) -> None:
        keys = frozenset({"r ravindra"})
        r = player_alias_resolve.resolve_player_to_history_key("Rachin Ravindra", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "r ravindra")

    def test_curated_alias_override_ruturaj_gaikwad(self) -> None:
        keys = frozenset({"rd gaikwad"})
        r = player_alias_resolve.resolve_player_to_history_key("Ruturaj Gaikwad", keys)
        self.assertEqual(r.resolution_type, "alias_match")
        self.assertEqual(r.resolved_history_key, "rd gaikwad")


if __name__ == "__main__":
    unittest.main()
