import numpy as np

from fdd.metrics import ap_metric, best_f1_threshold, detection_metrics, paired_block_bootstrap


def test_best_f1_threshold_separable():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr, f1 = best_f1_threshold(y, s)
    assert f1 == 1.0
    m = detection_metrics(y, s, thr)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 0)


def test_bootstrap_identical_methods_is_zero():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(int)
    s = rng.random(500) + y
    blocks = rng.integers(0, 25, 500)
    r = paired_block_bootstrap(y, blocks, [(s, 0.5)], [(s, 0.5)], ap_metric, 50, 0)
    assert r["difference"] == 0 and r["ci95_low"] == 0 and r["ci95_high"] == 0
