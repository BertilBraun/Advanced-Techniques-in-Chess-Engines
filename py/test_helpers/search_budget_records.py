from __future__ import annotations

import numpy as np
from src.search_budget.analysis_log import ANALYSIS_RECORD_DTYPE
from src.search_budget.policy import LOG_KL_EPSILON


def analysis_records(
    predicted: np.ndarray,
    target_log_kl: np.ndarray,
    top_visit_share: np.ndarray,
    policy_entropy: np.ndarray,
    ply: np.ndarray,
    baseline_visits: int = 400,
    source_generation: int = 12,
) -> np.ndarray:
    records = np.zeros(predicted.shape[0], dtype=ANALYSIS_RECORD_DTYPE)
    records['predicted_curve'] = predicted.astype(np.float32)
    records['policy_kl'] = (np.exp(target_log_kl) - LOG_KL_EPSILON).clip(min=0.0).astype(np.float32)
    records['top_visit_share'] = top_visit_share.astype(np.float32)
    records['policy_entropy'] = policy_entropy.astype(np.float32)
    records['ply'] = ply.astype(np.uint32)
    records['baseline_visits'] = baseline_visits
    records['source_generation'] = source_generation
    return records
