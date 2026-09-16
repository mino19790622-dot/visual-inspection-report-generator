"""Deterministic tests for the judge cross-check maths.

No network and no LLM: the judge itself is exercised in CI. What matters here
is that the agreement and gate-impact arithmetic is right, because a wrong
number in this file would be used to decide whether the README gate is sound.
"""
from __future__ import annotations

import pytest

from eval import judge_agreement as ja


def _flat(v: int) -> dict[str, int]:
    """All four dimensions set to the same score."""
    return {"scene_id": v, "safety": v, "domain_awareness": v, "structure": v}


def _img(s: int, f: int = 4, d: int = 4, t: int = 4) -> dict[str, int]:
    return {"scene_id": s, "safety": f, "domain_awareness": d, "structure": t}


class TestRanking:
    def test_distinct_values(self):
        assert ja._rank([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]

    def test_ties_share_the_average_rank(self):
        # Two values tied for ranks 1 and 2 -> both get 1.5.
        assert ja._rank([5.0, 5.0, 9.0]) == [1.5, 1.5, 3.0]

    def test_three_way_tie(self):
        assert ja._rank([2.0, 2.0, 2.0]) == [2.0, 2.0, 2.0]

    def test_spearman_identical_series_is_one(self):
        assert ja.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_spearman_reversed_series_is_minus_one(self):
        assert ja.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_spearman_undefined_when_one_side_is_constant(self):
        # Not 0.0: a flat series has no ordering to correlate with.
        assert ja.spearman([3, 3, 3], [1, 2, 3]) is None


class TestAgreement:
    def test_identical_scores_agree_perfectly(self):
        judge = {"a": _flat(4), "b": _flat(5)}
        hand = {"a": _flat(4), "b": _flat(5)}
        agg = ja.agreement(judge, hand)
        assert agg["n_pairs"] == 8
        assert agg["exact_agreement"] == 1.0
        assert agg["mean_abs_difference"] == 0.0
        assert agg["signed_mean_difference_judge_minus_rater"] == 0.0

    def test_lenient_judge_shows_a_positive_signed_difference(self):
        # The bias direction is what the gate depends on; agreement alone
        # would hide it. All four dimensions are 2 points apart.
        agg = ja.agreement({"a": _flat(5)}, {"a": _flat(3)})
        assert agg["signed_mean_difference_judge_minus_rater"] == 2.0
        assert agg["exact_agreement"] == 0.0
        # Off by 2 on every dimension is beyond the "close enough" band.
        assert agg["within_1_agreement"] == 0.0
        assert agg["mean_abs_difference"] == 2.0

    def test_within_one_counts_near_misses(self):
        agg = ja.agreement({"a": _flat(5)}, {"a": _flat(4)})
        assert agg["exact_agreement"] == 0.0
        assert agg["within_1_agreement"] == 1.0
        assert agg["mean_abs_difference"] == 1.0
        assert agg["signed_mean_difference_judge_minus_rater"] == 1.0

    def test_partial_dimension_delta_is_averaged_not_treated_as_all_dims(self):
        # Only scene_id moves: 3 of 4 dimensions still agree exactly.
        agg = ja.agreement({"a": _img(5)}, {"a": _img(4)})
        assert agg["exact_agreement"] == 0.75
        assert agg["n_pairs"] == 4
        assert agg["signed_mean_difference_judge_minus_rater"] == 0.25

    def test_images_without_hand_scores_are_excluded(self):
        judge = {"a": _flat(4), "b": _flat(4), "c": _flat(4)}
        hand = {"a": _flat(4)}
        agg = ja.agreement(judge, hand)
        assert agg["n_images_compared"] == 1
        assert agg["n_pairs"] == 4

    def test_partial_dimension_coverage_is_counted_per_dimension(self):
        judge = {"a": _flat(4), "b": _flat(5)}
        hand = {"a": _flat(4), "b": {"scene_id": 5}}
        agg = ja.agreement(judge, hand)
        assert agg["per_dimension"]["scene_id"]["n"] == 2
        assert agg["per_dimension"]["safety"]["n"] == 1


class TestGateImpact:
    def test_threshold_translates_to_the_judge_average_required(self):
        gate = ja.gate_impact({}, {}, 3.7)
        assert gate["judge_avg_required_to_pass"] == round((3.7 - 2.0) / 0.6, 3)

    def test_no_flips_when_raters_agree(self):
        assert ja.gate_impact({"a": _flat(4)}, {"a": _flat(4)}, 3.7)["n_flipped"] == 0

    def test_flip_when_judge_passes_and_rater_fails(self):
        # judge avg 3.0 >= 2.833 passes; rater avg 2.0 < 2.833 fails.
        gate = ja.gate_impact({"a": _flat(3)}, {"a": _flat(2)}, 3.7)
        assert gate["n_flipped"] == 1
        assert gate["images_flipped"][0]["judge_passes"] is True
        assert gate["images_flipped"][0]["rater_passes"] is False

    def test_no_flip_when_both_fail(self):
        assert ja.gate_impact({"a": _flat(1)}, {"a": _flat(2)}, 3.7)["n_flipped"] == 0

    def test_no_flip_when_both_pass(self):
        assert ja.gate_impact({"a": _flat(4)}, {"a": _flat(3)}, 3.7)["n_flipped"] == 0


class TestLoaders:
    def test_skipped_judge_rows_are_ignored(self, tmp_path):
        p = tmp_path / "eval.json"
        p.write_text(
            '{"results": [{"id": "a", "judge_raw": {"skipped": true}},'
            ' {"id": "b", "judge_raw": {"scene_id": 4, "safety": 4,'
            ' "domain_awareness": 4, "structure": 4}}]}', encoding="utf-8")
        scores, _ = ja.load_judge_scores(str(p))
        assert list(scores) == ["b"]

    def test_incomplete_judge_rows_are_ignored(self, tmp_path):
        p = tmp_path / "eval.json"
        p.write_text(
            '{"results": [{"id": "a", "judge_raw": {"scene_id": 4}}]}',
            encoding="utf-8")
        scores, _ = ja.load_judge_scores(str(p))
        assert scores == {}

    def test_hand_file_reads_rater_metadata(self, tmp_path):
        p = tmp_path / "hand.json"
        p.write_text(
            '{"rater": "me", "blind": true, "scores": {"a": {"scene_id": 4,'
            ' "safety": 4, "domain_awareness": 4, "structure": 4}}}',
            encoding="utf-8")
        scores, meta = ja.load_hand_scores(str(p))
        assert meta["rater"] == "me"
        assert meta["blind"] is True
        assert "a" in scores
