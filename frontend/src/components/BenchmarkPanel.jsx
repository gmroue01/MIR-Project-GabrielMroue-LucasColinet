import React, { useState, useEffect } from "react";
import { getIndexingMetrics } from "../api";
import styles from "./BenchmarkPanel.module.css";

const DESCRIPTOR_LABELS = {
  color_histogram:    "Histo. Couleur",
  mobilenet_arcface:  "MobileNet ArcFace",
  mobilenet_zeroshot: "MobileNet ZeroShot",
  resnet50_zeroshot:  "ResNet50 ZeroShot",
  vit_b16_zeroshot:   "ViT-B/16 ZeroShot",
  dinov2_supcon:      "DinoV2 SupCon",
  dinov2_zeroshot:    "DinoV2 ZeroShot",
  sift:               "SIFT",
};

function Bar({ value, max, color }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className={styles.barBg}>
      <div className={styles.barFill} style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

export default function BenchmarkPanel() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getIndexingMetrics()
      .then(setMetrics)
      .catch(() => setError("Index non disponible. Lancez d'abord l'indexeur."));
  }, []);

  if (error) return <div className={styles.error}>{error}</div>;
  if (!metrics) return <div className={styles.loading}>Chargement des métriques...</div>;

  const entries = Object.entries(metrics);
  const computedEntries = entries.filter(([, v]) => v.source !== "precomputed");
  const maxTime = computedEntries.length > 0
    ? Math.max(...computedEntries.map(([, v]) => v.indexing_time_s))
    : 1;
  const maxSize = Math.max(...entries.map(([, v]) => v.descriptor_size_mb));
  const maxAvg = Math.max(...entries.map(([, v]) => v.avg_search_time_s));

  return (
    <div className={styles.panel}>
      <h2 className={styles.title}>Benchmark des descripteurs</h2>
      <p className={styles.subtitle}>Métriques mesurées lors de l'indexation de {entries[0]?.[1]?.num_images} images.</p>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Descripteur</th>
              <th>Dimension</th>
              <th>Temps indexation</th>
              <th></th>
              <th>Taille (MB)</th>
              <th></th>
              <th>Temps moy./img</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([key, m]) => (
              <tr key={key}>
                <td className={styles.nameCell}>{DESCRIPTOR_LABELS[key] || key}</td>
                <td className={styles.numCell}>{m.descriptor_dim}</td>
                <td className={styles.numCell}>
                  {m.source === "precomputed" ? "—" : `${m.indexing_time_s.toFixed(1)} s`}
                </td>
                <td className={styles.barCell}>
                  {m.source !== "precomputed" && <Bar value={m.indexing_time_s} max={maxTime} color="#60a5fa" />}
                </td>
                <td className={styles.numCell}>{m.descriptor_size_mb.toFixed(2)}</td>
                <td className={styles.barCell}>
                  <Bar value={m.descriptor_size_mb} max={maxSize} color="#34d399" />
                </td>
                <td className={styles.numCell}>{(m.avg_search_time_s * 1000).toFixed(1)} ms</td>
                <td className={styles.barCell}>
                  <Bar value={m.avg_search_time_s} max={maxAvg} color="#f59e0b" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className={styles.cards}>
        {entries.map(([key, m]) => (
          <div key={key} className={styles.card}>
            <h3 className={styles.cardTitle}>{DESCRIPTOR_LABELS[key] || key}</h3>
            <div className={styles.stat}>
              <span className={styles.statLabel}>Indexation</span>
              <span className={styles.statValue} style={{ color: "#60a5fa" }}>
                {m.source === "precomputed" ? "pré-calculé" : `${m.indexing_time_s.toFixed(1)} s`}
              </span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statLabel}>Taille</span>
              <span className={styles.statValue} style={{ color: "#34d399" }}>{m.descriptor_size_mb.toFixed(2)} MB</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statLabel}>Moy./image</span>
              <span className={styles.statValue} style={{ color: "#f59e0b" }}>{(m.avg_search_time_s * 1000).toFixed(2)} ms</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statLabel}>Dimension</span>
              <span className={styles.statValue}>{m.descriptor_dim}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
