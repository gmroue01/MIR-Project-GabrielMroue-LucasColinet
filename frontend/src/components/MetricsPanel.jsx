import React from "react";
import styles from "./MetricsPanel.module.css";

function MetricRow({ label, value, asPercent = true }) {
  const display = asPercent
    ? `${(value * 100).toFixed(1)}%`
    : typeof value === "number" ? value.toFixed(4) : value;
  return (
    <div className={styles.row}>
      <span className={styles.rowLabel}>{label}</span>
      <span className={styles.rowValue}>{display}</span>
    </div>
  );
}

export default function MetricsPanel({ metrics, searchTime }) {
  if (!metrics) return null;
  const { top20, top50, total_relevant_in_db } = metrics;

  return (
    <div className={styles.panel}>
      <h3 className={styles.title}>Métriques d'évaluation</h3>

      <div className={styles.group}>
        <span className={styles.groupLabel}>Top 20</span>
        <MetricRow label="Precision" value={top20.precision} />
        <MetricRow label="Recall" value={top20.recall} />
        <MetricRow label="Avg. Precision" value={top20.average_precision} />
        <MetricRow label="R-Precision" value={top20.r_precision} />
      </div>

      <div className={styles.group}>
        <span className={styles.groupLabel}>Top 50</span>
        <MetricRow label="Precision" value={top50.precision} />
        <MetricRow label="Recall" value={top50.recall} />
        <MetricRow label="Avg. Precision" value={top50.average_precision} />
        <MetricRow label="R-Precision" value={top50.r_precision} />
      </div>

      <div className={styles.footer}>
        <span>Pertinents dans la BDD : <strong>{total_relevant_in_db}</strong></span>
        <span>Temps de recherche : <strong>{(searchTime * 1000).toFixed(0)} ms</strong></span>
      </div>
    </div>
  );
}
