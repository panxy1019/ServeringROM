#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a sealed ServingROM snapshot run.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()
    import numpy as np
    import pyarrow.parquet as pq

    directory = args.run_root / "derived" / "snapshots"
    windows = pq.read_table(directory / "window_table.parquet").to_pylist()
    state = np.load(directory / "full_state.npy")
    disturbance = np.load(directory / "disturbance.npy")
    output = np.load(directory / "output.npy")
    index = args.window
    if not 0 <= index < len(windows):
        raise SystemExit(f"window must be in [0,{len(windows) - 1}]")
    print(json.dumps({
        "window": windows[index],
        "state_shape": list(state.shape),
        "disturbance_shape": list(disturbance.shape),
        "output_shape": list(output.shape),
        "state_nonzero": int(np.count_nonzero(state[index])),
        "disturbance_nonzero": int(np.count_nonzero(disturbance[index])),
        "output_nonzero": int(np.count_nonzero(output[index])),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
