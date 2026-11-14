import numpy as np
import pytest

from evaluation.coverage_accuracy import coverage_accuracy, load_calibrator


def test_coverage_accuracy_hand_example():
    # scores sorted high->low: 0.9(1) 0.8(0) 0.3(1) 0.1(0)
    cov, acc = coverage_accuracy(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.8, 0.3]))
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert acc.tolist() == [1.0, 0.5, 2 / 3, 0.5]


@pytest.mark.parametrize("dataset", ["movies", "songs", "basketball"])
def test_numpy_forward_reproduces_p2_test_alignment(dataset):
    from calibrator.data_prep import load_split

    predict, meta = load_calibrator("qwen2.5-7b", dataset)
    try:
        split = load_split("qwen2.5-7b", dataset, protocol=meta["protocol"], features=meta["features"])
    except FileNotFoundError:
        pytest.skip("paper snapshot not available")
    p = predict(split.X_test)
    assert np.mean((p >= meta["threshold"]) == split.y_test) == pytest.approx(meta["test_metrics"]["alignment"], abs=1e-9)
