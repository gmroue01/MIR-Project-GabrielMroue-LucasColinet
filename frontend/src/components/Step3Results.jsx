import React, { useState } from "react";
import styles from "./Step3Results.module.css";

export default function Step3Results({ result, onNext, onNewSearch, onRerank, reranking, topK = 50 }) {
  const [poolPercent, setPoolPercent] = useState(25);

  if (!result) return null;

  const results  = topK === 20 ? result.results_top20 : result.results_top50;
  const relevant = results.filter(r => r.class === result.query_class).length;

  const poolSize = topK + Math.ceil(topK * poolPercent / 100);

  const handleRerank = () => onRerank(poolPercent);

  return (
    <div className={styles.step}>
      {/* Header */}
      <div className={styles.head}>
        <div className={styles.queryBox}>
          <img src={`/images/${result.query_filename}`} alt="query" className={styles.queryImg} />
          <div>
            <span className={styles.queryLbl}>Requête</span>
            <span className={styles.queryClass}>{result.query_class.replace(/_/g, " ")}</span>
          </div>
        </div>

        <div className={styles.headCenter}>
          <div className={styles.stepId}>03</div>
          <div>
            <h2 className={styles.title}>Résultats de recherche</h2>
            <p className={styles.sub}>
              <span className={styles.statRelevant}>{relevant} pertinents</span>
              <span className={styles.statSep}>/</span>
              <span className={styles.statTotal}>{results.length} résultats</span>
              {result.reranked && <span className={styles.rerankBadge}>✦ SIFT-RANSAC</span>}
            </p>
          </div>
        </div>

        <div className={styles.headRight}>
          <span className={styles.topKBadge}>Top {topK}</span>
          <button className={styles.newSearchBtn} onClick={onNewSearch}>↺ Nouvelle</button>
        </div>
      </div>

      {/* Grid */}
      <div className={styles.body}>
        <div className={styles.grid}>
          {results.map(r => {
            const ok = r.class === result.query_class;
            return (
              <div
                key={r.index}
                className={`${styles.card} ${ok ? styles.cardOk : styles.cardKo}`}
                title={`${r.class.replace(/_/g, " ")}\nd = ${r.distance.toFixed(4)}${r.sift_score !== undefined ? `\nSIFT inliers: ${r.sift_score}` : ""}`}
              >
                <span className={styles.rank}>#{r.rank}</span>
                {r.sift_score !== undefined && r.sift_score > 0 && (
                  <span className={styles.siftScore}>{r.sift_score}</span>
                )}
                <img src={`/images/${r.filename}`} alt={r.filename} loading="lazy" className={styles.thumb} />
                <span className={`${styles.dot} ${ok ? styles.dotOk : styles.dotKo}`} />
              </div>
            );
          })}
        </div>

        {/* SIFT-RANSAC Reranking panel */}
        <div className={styles.rerankPanel}>
          <div className={styles.rerankHeader}>
            <span className={styles.rerankIcon}>⬡</span>
            <div>
              <span className={styles.rerankTitle}>Reranking SIFT-RANSAC</span>
              <span className={styles.rerankDesc}>
                Affine les résultats en vérifiant la cohérence géométrique des correspondances
              </span>
            </div>
            {result.reranked && (
              <span className={styles.rerankApplied}>✓ Appliqué</span>
            )}
          </div>

          <div className={styles.rerankControls}>
            <div className={styles.sliderGroup}>
              <div className={styles.sliderLabels}>
                <span className={styles.sliderLabel}>Pool de candidats</span>
                <span className={styles.sliderValue}>
                  top-{topK} + {poolPercent}% = <strong>{poolSize}</strong> candidats évalués
                </span>
              </div>
              <div className={styles.sliderRow}>
                <span className={styles.sliderMin}>10%</span>
                <input
                  type="range"
                  min={10}
                  max={100}
                  step={5}
                  value={poolPercent}
                  onChange={e => setPoolPercent(Number(e.target.value))}
                  className={styles.slider}
                  style={{ "--pct": `${((poolPercent - 10) / 90) * 100}%` }}
                />
                <span className={styles.sliderMax}>100%</span>
              </div>
            </div>

            <button
              className={`${styles.rerankBtn} ${reranking ? styles.rerankBtnLoading : ""}`}
              onClick={handleRerank}
              disabled={reranking}
            >
              {reranking ? (
                <><span className={styles.spinner} /> Reranking…</>
              ) : (
                <>⬡ Lancer le reranking</>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className={styles.foot}>
        <div className={styles.legend}>
          <span className={styles.legendDotOk} /> Pertinent
          <span className={styles.legendDotKo} style={{ marginLeft: "1.5rem" }} /> Non pertinent
          {result.reranked && (
            <span className={styles.legendSift} style={{ marginLeft: "1.5rem" }}>
              <span className={styles.siftScoreDot} /> Inliers SIFT
            </span>
          )}
        </div>
        <button className={styles.nextBtn} onClick={onNext}>
          Voir les métriques <span>→</span>
        </button>
      </div>
    </div>
  );
}
