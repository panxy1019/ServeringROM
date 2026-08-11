from __future__ import annotations

import numpy as np

from servingrom_control_modeling.redesign import (
    _descriptor_manifest,
    _extract_descriptors,
    _scheme2_pairs,
)


def _index(names: list[str]) -> list[dict[str, object]]:
    return [
        {"index": index, "name": name, "block": "test", "unit": "requests"}
        for index, name in enumerate(names)
    ]


def test_descriptor_pairing_requires_exact_counterpart() -> None:
    rows = _index([
        "decode_d1_running_count", "decode_d2_running_count",
        "decode_d1_waiting_count", "decode_d2_waiting_count",
        "decode_d1_expected_remaining_tokens", "decode_d2_expected_remaining_tokens",
        "decode_route_imbalance_requests", "decode_route_imbalance_tokens",
        "decode-0.running_count.context_0.progress_0",
        "decode-1.running_count.context_0.progress_0",
        "decode-0.running_count.context_1.progress_0",
    ])
    result = _descriptor_manifest(rows)
    binned = result["objects"]["selected_binned"]
    running = next(row for row in binned if row.name == "binned_running_count_imbalance")
    assert len(running.left) == 1
    assert len(running.right) == 1


def test_core_differentials_are_signed_a_minus_b() -> None:
    rows = _index([
        "decode_d1_running_count", "decode_d2_running_count",
        "decode_d1_waiting_count", "decode_d2_waiting_count",
        "decode_d1_expected_remaining_tokens", "decode_d2_expected_remaining_tokens",
    ])
    descriptors = _descriptor_manifest(rows)["objects"]["core3"]
    value = _extract_descriptors(np.asarray([[7.0, 2.0, 3.0, 5.0, 19.0, 4.0]]), descriptors)
    assert value.tolist() == [[5.0, -2.0, 15.0]]


def test_scheme2_pairs_keep_core_first_and_ignore_unmatched_bins() -> None:
    rows = _index([
        "decode_d1_running_count", "decode_d2_running_count",
        "decode_d1_waiting_count", "decode_d2_waiting_count",
        "decode_d1_expected_remaining_tokens", "decode_d2_expected_remaining_tokens",
        "decode-0.running_count.context_0.progress_0",
        "decode-1.running_count.context_0.progress_0",
        "decode-0.running_count.context_1.progress_0",
    ])
    pairs = _scheme2_pairs(rows)
    assert [row.name for row in pairs[:3]] == [
        "running_imbalance", "waiting_imbalance", "remaining_token_imbalance",
    ]
    assert len(pairs) == 4
