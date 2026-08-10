from __future__ import annotations

from typing import Mapping


def weighted_fair_decoder(
    decoder_a: str,
    decoder_b: str,
    rho_a: float,
    assignment_counts: Mapping[str, int],
) -> tuple[str, dict[str, float]]:
    """Choose the largest cumulative assignment deficit deterministically."""
    total_after = int(assignment_counts.get(decoder_a, 0)) + int(
        assignment_counts.get(decoder_b, 0)
    ) + 1
    deficits = {
        decoder_a: rho_a * total_after - int(assignment_counts.get(decoder_a, 0)),
        decoder_b: (1.0 - rho_a) * total_after - int(assignment_counts.get(decoder_b, 0)),
    }
    if deficits[decoder_a] == deficits[decoder_b]:
        return decoder_a, deficits
    return max((decoder_a, decoder_b), key=lambda key: deficits[key]), deficits
