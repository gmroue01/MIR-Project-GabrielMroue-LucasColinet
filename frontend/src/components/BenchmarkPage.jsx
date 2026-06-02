import React, { useState, useEffect } from "react";
import { getIndexingMetrics } from "../api";
import styles from "./BenchmarkPage.module.css";

const LABELS = {
  color_histogram:    "Histo. Couleur",
  mobilenet_arcface:  "MobileNet ArcFace",
  mobilenet_zeroshot: "MobileNet ZeroShot",
  resnet50_zeroshot:  "ResNet50 ZeroShot",
  vit_b16_zeroshot:   "ViT-B/16 ZeroShot",
  dinov2_supcon:      "DinoV2 SupCon",
  dinov2_zeroshot:    "DinoV2 ZeroShot",
  sift:               "SIFT",
};

const COLORS = {
  color_histogram:    "#8b5cf6",
  mobilenet_arcface:  "#fb923c",
  mobilenet_zeroshot: "#fcd34d",
  resnet50_zeroshot:  "#6ee7b7",
  vit_b16_zeroshot:   "#f9a8d4",
  dinov2_supcon:      "#818cf8",
  dinov2_zeroshot:    "#a5b4fc",
  sift:               "#a78bfa",
};

function formatTime(s) {
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  if (s >= 60)   return `${Math.floor(s / 60)} min`;
  return `${s.toFixed(1)} s`;
}

// Log-scale bar so small and large values are both visible
function Bar({ value, max, color }) {
  const logVal = value > 0 ? Math.log1p(value) : 0;
  const logMax = max  > 0 ? Math.log1p(max)   : 1;
  const pct = logMax > 0 ? Math.min(100, (logVal / logMax) * 100) : 0;
  return (
    <div className={styles.bar}>
      <div className={styles.barFill} style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

export default function BenchmarkPage() {
  const [metrics, setMetrics] = useState(null);
  const [error,   setError]   = useState("");

  useEffect(() => {
    getIndexingMetrics()
      .then(setMetrics)
      .catch(() => setError("Index non disponible — lancez d'abord l'indexeur."));
  }, []);

  if (error) return (
    <div className={styles.page}>
      <div className={styles.errorBox}>{error}</div>
    </div>
  );

  if (!metrics) return (
    <div className={styles.page}>
      <div className={styles.loading}>
        <div className={styles.orbit}><div className={styles.orbitDot} /></div>
        <span>Chargement des métriques…</span>
      </div>
    </div>
  );

  const entries   = Object.entries(metrics);
  const numImages = entries[0]?.[1]?.num_images ?? "?";
  const maxSize   = Math.max(...entries.map(([, v]) => v.descriptor_size_mb));
  const maxAvg    = Math.max(...entries.map(([, v]) => v.avg_search_time_s));
  const maxDim    = Math.max(...entries.map(([, v]) => v.original_dim ?? v.descriptor_dim));
  const maxTime   = Math.max(...entries.map(([, v]) => v.indexing_time_s));

  return (
    <div className={styles.page}>
      <div className={styles.pageHead}>
        <h1 className={styles.pageTitle}><span className={styles.star}>✦</span> Benchmark des descripteurs</h1>
        <p className={styles.pageSub}>
          Métriques mesurées sur {numImages} images ·{" "}
          <span className={styles.precomputedNote}>Temps d'indexation mesurés sur GPU T4 (Google Colab)</span>
        </p>
      </div>

      {/* Cards grid */}
      <div className={styles.cards}>
        {entries.map(([key, m]) => {
          const color       = COLORS[key] || "#8b5cf6";
          const precomputed = m.source === "precomputed";
          return (
            <div key={key} className={styles.card} style={{ "--c": color }}>
              <div className={styles.cardHead}>
                <span className={styles.cardDot} />
                <span className={styles.cardName}>{LABELS[key] || key}</span>
                <div className={styles.cardMeta}>
                  <span className={styles.cardDim}>
                    {m.original_dim ? `${m.original_dim}→${m.descriptor_dim}D` : `${m.descriptor_dim}D`}
                  </span>
                  {precomputed && <span className={styles.preTag}>Colab</span>}
                </div>
              </div>

              <div className={styles.stats}>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Indexation</span>
                  <span className={styles.statVal} style={{ color }}>{formatTime(m.indexing_time_s)}</span>
                  <Bar value={m.indexing_time_s} max={maxTime} color={color} />
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Taille index</span>
                  <span className={styles.statVal} style={{ color }}>{m.descriptor_size_mb.toFixed(2)} MB</span>
                  <Bar value={m.descriptor_size_mb} max={maxSize} color={color} />
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Temps recherche / img</span>
                  <span className={styles.statVal} style={{ color }}>{(m.avg_search_time_s * 1000).toFixed(2)} ms</span>
                  <Bar value={m.avg_search_time_s} max={maxAvg} color={color} />
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Dimension</span>
                  <span className={styles.statVal} style={{ color }}>
                    {m.original_dim ? `${m.original_dim} → ${m.descriptor_dim}` : m.descriptor_dim}
                  </span>
                  <Bar value={m.original_dim ?? m.descriptor_dim} max={maxDim} color={color} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparison table */}
      <div className={styles.tableSection}>
        <h2 className={styles.sectionTitle}>Vue comparative</h2>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Descripteur</th>
                <th>Dim.</th>
                <th>Source</th>
                <th>Indexation</th>
                <th>Taille (MB)</th>
                <th className={styles.barCol}></th>
                <th>Recherche</th>
                <th className={styles.barCol}></th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([key, m]) => {
                const color       = COLORS[key] || "#8b5cf6";
                const precomputed = m.source === "precomputed";
                return (
                  <tr key={key}>
                    <td>
                      <span className={styles.tdDot} style={{ background: color }} />
                      {LABELS[key] || key}
                    </td>
                    <td className={styles.num}>
                      {m.original_dim ? `${m.original_dim}→${m.descriptor_dim}` : m.descriptor_dim}
                    </td>
                    <td>
                      {precomputed
                        ? <span className={styles.preTag}>Colab</span>
                        : <span className={styles.localTag}>Local</span>}
                    </td>
                    <td className={styles.num}>{formatTime(m.indexing_time_s)}</td>
                    <td className={styles.num}>{m.descriptor_size_mb.toFixed(2)}</td>
                    <td className={styles.barCell}><Bar value={m.descriptor_size_mb} max={maxSize} color={color} /></td>
                    <td className={styles.num}>{(m.avg_search_time_s * 1000).toFixed(2)} ms</td>
                    <td className={styles.barCell}><Bar value={m.avg_search_time_s} max={maxAvg} color={color} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
