from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BIN = ROOT / "bin" / "step3_mm"
STEP1_FEATURES = ROOT / "output" / "step1_features" / "feature_table_1s.parquet"
STEP2_FEATURES_CSV = ROOT / "output" / "step2_hawkes" / "feature_table_step2_1s.csv"
STEP2_DIR = ROOT / "output" / "step2_hawkes"


def require(path: Path, hint: str = "run `make build`") -> Path:
    if not path.exists():
        pytest.skip(f"missing {path.relative_to(ROOT)} ({hint})")
    return path


@pytest.fixture(scope="session")
def engine_bin() -> Path:
    return require(BIN)
