"""Late fusion of the (provided) imaging probability and the (re-implemented) EHR probability.

Two parameter-free operators, matching the reference benchmark:
  - average : arithmetic mean of the two probabilities
  - maximum : elementwise maximum

The imaging probability is always the RadFusion-provided vision-model score (Vision.csv `pred`).
"""
from __future__ import annotations
import numpy as np


def average_fusion(imaging_prob: np.ndarray, ehr_prob: np.ndarray) -> np.ndarray:
    return np.mean(np.vstack([imaging_prob, ehr_prob]), axis=0)


def maximum_fusion(imaging_prob: np.ndarray, ehr_prob: np.ndarray) -> np.ndarray:
    return np.maximum(imaging_prob, ehr_prob)
