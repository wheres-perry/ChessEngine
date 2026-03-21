from __future__ import annotations

import random

from elo_tests.models import ScheduledGame


def build_paired_schedule(
    openings: list[str],
    paired_blocks_target: int,
    seed: int,
) -> list[ScheduledGame]:
    if not openings:
        raise ValueError("openings list must not be empty")

    rng = random.Random(seed)
    indexed_openings = list(enumerate(openings))
    rng.shuffle(indexed_openings)

    schedule: list[ScheduledGame] = []
    for pair_id in range(paired_blocks_target):
        opening_id, opening_fen = indexed_openings[pair_id % len(indexed_openings)]
        first_seed = seed + pair_id * 2
        second_seed = seed + pair_id * 2 + 1

        schedule.append(
            ScheduledGame(
                pair_id=pair_id,
                opening_id=opening_id,
                opening_fen=opening_fen,
                game_index_in_pair=0,
                seed=first_seed,
                candidate_is_white=True,
            )
        )
        schedule.append(
            ScheduledGame(
                pair_id=pair_id,
                opening_id=opening_id,
                opening_fen=opening_fen,
                game_index_in_pair=1,
                seed=second_seed,
                candidate_is_white=False,
            )
        )

    return schedule
