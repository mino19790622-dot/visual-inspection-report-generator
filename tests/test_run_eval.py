"""Deterministic tests for the golden-set runner's selection and record shape.

No network and no LLM: the judge, the VLM and the detector are all exercised
elsewhere. What is covered here is the part that decides *what* gets evaluated
and *what* gets written down -- both of which have silently-wrong failure modes
(a subset that selects nothing, a record that cannot be audited).
"""
from __future__ import annotations

from eval.run_eval import select_items

ITEMS = [{"id": n} for n in ("alpha", "beta", "gamma", "delta")]


class TestSelectItems:
    def test_no_spec_selects_everything(self):
        got, missing = select_items(ITEMS, None)
        assert got == ITEMS
        assert missing == []

    def test_empty_spec_is_not_an_empty_selection(self):
        # An empty string must mean "all", never "none": selecting nothing
        # would make the gate pass vacuously.
        got, missing = select_items(ITEMS, "")
        assert got == ITEMS
        assert missing == []

    def test_single_id(self):
        got, _ = select_items(ITEMS, "beta")
        assert [i["id"] for i in got] == ["beta"]

    def test_multiple_ids(self):
        got, _ = select_items(ITEMS, "beta,delta")
        assert [i["id"] for i in got] == ["beta", "delta"]

    def test_whitespace_is_tolerated(self):
        got, _ = select_items(ITEMS, " beta , delta ")
        assert [i["id"] for i in got] == ["beta", "delta"]

    def test_order_follows_the_golden_set_not_the_spec(self):
        # Keeps run-to-run output diffable regardless of how ids were typed.
        got, _ = select_items(ITEMS, "delta,beta")
        assert [i["id"] for i in got] == ["beta", "delta"]

    def test_unknown_ids_are_reported_not_dropped_silently(self):
        got, missing = select_items(ITEMS, "beta,nope")
        assert [i["id"] for i in got] == ["beta"]
        assert missing == ["nope"]

    def test_all_unknown_yields_nothing_and_reports_everything(self):
        got, missing = select_items(ITEMS, "x,y")
        assert got == []
        assert missing == ["x", "y"]

    def test_duplicate_ids_do_not_duplicate_items(self):
        got, _ = select_items(ITEMS, "beta,beta")
        assert [i["id"] for i in got] == ["beta"]

    def test_blank_entries_are_ignored(self):
        got, missing = select_items(ITEMS, "beta,,delta")
        assert [i["id"] for i in got] == ["beta", "delta"]
        assert missing == []
