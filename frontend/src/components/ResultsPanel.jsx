import React, { useState } from "react";
import styles from "./ResultsPanel.module.css";

export default function ResultsPanel({ result, onBack, onSelectNew }) {
  const [view, setView] = useState(50);
  const results = view === 20 ? result.results_top20 : result.results_top50;

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.queryBox}>
          <img
            src={`/images/${result.query_filename}`}
            alt="query"
            className={styles.queryImg}
          />
          <div className={styles.queryInfo}>
            <span className={styles.queryLabel}>Image requête</span>
            <span className={styles.queryClass}>{result.query_class.replace(/_/g, " ")}</span>
            <span className={styles.queryFile}>{result.query_filename}</span>
          </div>
        </div>

        <div className={styles.actions}>
          <div className={styles.viewToggle}>
            {[20, 50].map((k) => (
              <button
                key={k}
                className={`${styles.toggleBtn} ${view === k ? styles.toggleActive : ""}`}
                onClick={() => setView(k)}
              >
                Top {k}
              </button>
            ))}
          </div>
          <button className={styles.backBtn} onClick={onBack}>
            ← Nouvelle recherche
          </button>
        </div>
      </div>

      <div className={styles.grid}>
        {results.map((r) => {
          const relevant = r.class === result.query_class;
          return (
            <button
              key={r.index}
              className={`${styles.card} ${relevant ? styles.relevant : styles.irrelevant}`}
              onClick={() => onSelectNew({ index: r.index, filename: r.filename, class: r.class })}
              title={`Rang ${r.rank} — ${r.class.replace(/_/g, " ")}\nd = ${r.distance.toFixed(4)}`}
            >
              <span className={styles.rank}>#{r.rank}</span>
              <img
                src={`/images/${r.filename}`}
                alt={r.filename}
                loading="lazy"
                className={styles.thumb}
              />
              <span className={`${styles.badge} ${relevant ? styles.badgeOk : styles.badgeKo}`}>
                {relevant ? "✓" : "✗"}
              </span>
              <span className={styles.dist}>{r.distance.toFixed(4)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
