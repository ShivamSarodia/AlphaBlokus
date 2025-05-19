"""
Find the first file (in lexicographic order) such that the **cumulative**
game-count of all files that come *before* it is ≥ 6 000 000.

Assumes filenames look like “…_<game-count>.npz”.
"""

from pathlib import Path
import re

DIR = Path("/Users/shivamsarodia/Dev/BlokusBot/data/2024-12-30_23-23-24-rubefaction/games")
THRESHOLD = 7_500_000

_game_re = re.compile(r"_(\d+)\.npz$")

def game_count(fname: str) -> int:
    m = _game_re.search(fname)
    if not m:
        raise ValueError(f"Bad filename format: {fname}")
    return int(m.group(1))

def first_after_threshold(directory: Path, threshold: int = THRESHOLD):
    cum = 0
    for f in sorted(directory.glob("*.npz")):          # alphabetical order
        if cum >= threshold:                           # only files *before* f
            return f
        cum += game_count(f.name)
    return None                                        # never reached

if __name__ == "__main__":
    result = first_after_threshold(DIR, THRESHOLD)
    if result is None:
        print("No file qualifies; cumulative never reached the threshold.")
    else:
        print(result)
