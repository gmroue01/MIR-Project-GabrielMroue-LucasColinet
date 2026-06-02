"""
Per-query evaluation metrics for image retrieval.
Relevance = same class (brand + model) as the query image.
"""
import numpy as np
from typing import List


def _relevance(results: List[dict], query_class: str) -> List[int]:
    """Binary relevance vector for the ranked result list."""
    return [1 if r["class"] == query_class else 0 for r in results]


def precision_at_k(results: List[dict], query_class: str, k: int) -> float:
    rel = _relevance(results[:k], query_class)
    return float(sum(rel) / k) if k > 0 else 0.0


def recall_at_k(results: List[dict], query_class: str, k: int, total_relevant: int) -> float:
    rel = _relevance(results[:k], query_class)
    return float(sum(rel) / total_relevant) if total_relevant > 0 else 0.0


def average_precision(results: List[dict], query_class: str, total_relevant: int) -> float:
    """Average Precision (AP) over all retrieved results."""
    rel = _relevance(results, query_class)
    if total_relevant == 0:
        return 0.0
    running_sum = 0.0
    relevant_seen = 0
    for i, r in enumerate(rel, start=1):
        if r == 1:
            relevant_seen += 1
            running_sum += relevant_seen / i
    return float(running_sum / total_relevant)


def r_precision(results: List[dict], query_class: str, total_relevant: int) -> float:
    """Precision at rank R, where R = number of relevant documents in collection."""
    if total_relevant == 0:
        return 0.0
    rel = _relevance(results[:total_relevant], query_class)
    return float(sum(rel) / total_relevant)


def compute_all(
    results: List[dict],
    query_class: str,
    total_relevant: int,
    k_values: List[int] = None,
) -> dict:
    if k_values is None:
        k_values = [20, 50]

    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(results, query_class, k)
        metrics[f"recall@{k}"] = recall_at_k(results, query_class, k, total_relevant)

    metrics["average_precision"] = average_precision(results, query_class, total_relevant)
    metrics["r_precision"] = r_precision(results, query_class, total_relevant)

    # MAP is computed over a set of queries; here we return AP for this single query.
    # The API endpoint that aggregates MAP over multiple queries uses this AP value.
    metrics["ap"] = metrics["average_precision"]

    return metrics
