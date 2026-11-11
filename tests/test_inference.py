"""Raw-input path: a bundled sample file must reproduce the score recorded in the answer key; bad shapes are rejected."""
import os
import numpy as np
import pandas as pd
import pytest
from src.inference import engine as E

SAMPLE = "testing_data/samples/sample_03.csv"


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample files not generated")
def test_sample_score_matches_key():
    key = pd.read_csv("testing_data/answer_key_and_results.csv").set_index("file")
    r = E.analyze(np.loadtxt(SAMPLE, delimiter=","))
    assert abs(r["subject_score"] - key.loc["sample_03.csv", "subject_score"]) < 1e-2
    assert len(r["seed_max"]) == 20


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        E.as_timeseries(np.zeros((100, 10)))
    with pytest.raises(ValueError):
        E.as_timeseries(np.full((100, 360), np.nan))
    with pytest.raises(ValueError):
        E.as_timeseries(np.zeros((20, 360)))
    assert E.as_timeseries(np.random.randn(360, 100)).shape == (100, 360)  # transpose accepted
