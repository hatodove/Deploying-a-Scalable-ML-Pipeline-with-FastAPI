from pathlib import Path


def test_census_exists():
    assert Path("data/census.csv").exists()
