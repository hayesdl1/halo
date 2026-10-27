import pytest

from halo import METRICS
from halo.config import config_hash, load_config


def test_default_config_loads():
    cfg = load_config()
    assert tuple(cfg["score"]["metrics"]) == METRICS
    assert cfg["universe"]["n_stocks"] == 1000
    assert cfg["fundamentals"]["lag_months"] == 6
    assert cfg["holdout"]["start"] == "2023-01"
    assert cfg["portfolio"]["cost_bps"] == 10


def test_overrides_merge_nested():
    cfg = load_config(overrides={"universe": {"n_stocks": 500}})
    assert cfg["universe"]["n_stocks"] == 500
    assert cfg["universe"]["min_price"] == 5.0


@pytest.mark.parametrize("override", [
    {"score": {"metrics": ["tai", "bogus"]}},
    {"fundamentals": {"ppe": "ppex"}},
    {"fundamentals": {"nonpositive_ebitda": "zero"}},
    {"tuning": {"first_train_end": "2023-06"}},
    {"tuning": {"walkforward_years": [2008, 2023]}},
])
def test_invalid_config_rejected(override):
    with pytest.raises(AssertionError):
        load_config(overrides=override)


def test_hash_stable_and_sensitive():
    a, b = load_config(), load_config()
    assert config_hash(a) == config_hash(b)
    c = load_config(overrides={"fundamentals": {"lag_months": 3}})
    assert config_hash(a) != config_hash(c)
