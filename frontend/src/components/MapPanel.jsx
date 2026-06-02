import React, { useState } from "react";
import { computeMap } from "../api";
import styles from "./MapPanel.module.css";

const DESCRIPTORS = [
  { id: "color_histogram",    label: "Histo. Couleur" },
  { id: "mobilenet_arcface",  label: "MobileNet ArcFace" },
  { id: "mobilenet_zeroshot", label: "MobileNet ZeroShot" },
  { id: "resnet50_zeroshot",  label: "ResNet50 ZeroShot" },
  { id: "vit_b16_zeroshot",   label: "ViT-B/16 ZeroShot" },
  { id: "dinov2_supcon",      label: "DinoV2 SupCon" },
  { id: "dinov2_zeroshot",    label: "DinoV2 ZeroShot" },
  { id: "sift",               label: "SIFT" },
];

const MEASURES = [
  { id: "euclidean", label: "Euclidienne" },
  { id: "cosine",    label: "Cosinus" },
];

export default function MapPanel() {
  const [selectedDescriptors, setSelectedDescriptors] = useState(["color_histogram"]);
  const [measure, setMeasure] = useState("euclidean");
  const [topK, setTopK] = useState(50);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const toggleDescriptor = (id) => {
    setSelectedDescriptors((prev) => {
      const next = prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id];
      return next.length === 0 ? prev : next;
    });
  };

  const handleCompute = async () => {
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const data = await computeMap(selectedDescriptors, measure, topK, 46);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || "Erreur lors du calcul MAP.");
    } finally {
      setLoading(false);
    }
  };

  const entries = result ? Object.entries(result.ap_per_class).sort(([, a], [, b]) => b - a) : [];

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h2 className={styles.title}>Mean Average Precision (MAP)</h2>
        <p className={styles.subtitle}>
          Calcule la MAP sur 1 requête par classe (46 classes), pour la combinaison choisie.
        </p>
      </div>

      <div className={styles.controls}>
        <div className={styles.controlGroup}>
          <label className={styles.label}>Descripteurs</label>
          <div className={styles.chips}>
            {DESCRIPTORS.map((d) => (
              <button
                key={d.id}
                className={`${styles.chip} ${selectedDescriptors.includes(d.id) ? styles.chipActive : ""}`}
                onClick={() => toggleDescriptor(d.id)}
              >
                {d.label}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlGroup}>
          <label className={styles.label}>Mesure</label>
          <div className={styles.chips}>
            {MEASURES.map((m) => (
              <button
                key={m.id}
                className={`${styles.chip} ${measure === m.id ? styles.chipActive : ""}`}
                onClick={() => setMeasure(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlGroup}>
          <label className={styles.label}>Top-K</label>
          <div className={styles.chips}>
            {[20, 50].map((k) => (
              <button
                key={k}
                className={`${styles.chip} ${topK === k ? styles.chipActive : ""}`}
                onClick={() => setTopK(k)}
              >
                Top {k}
              </button>
            ))}
          </div>
        </div>

        {error && <p className={styles.error}>{error}</p>}

        <button className={styles.btn} onClick={handleCompute} disabled={loading}>
          {loading ? "Calcul en cours..." : "Calculer MAP"}
        </button>
      </div>

      {result && (
        <div className={styles.results}>
          <div className={styles.mapScore}>
            <span className={styles.mapLabel}>MAP@{topK}</span>
            <span className={styles.mapValue}>{(result.map * 100).toFixed(2)}%</span>
            <span className={styles.mapMeta}>
              {result.num_queries} requêtes · {result.elapsed_s}s
            </span>
          </div>

          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Classe</th>
                  <th>AP</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {entries.map(([cls, ap]) => (
                  <tr key={cls}>
                    <td className={styles.clsCell}>{cls.replace(/_/g, " ")}</td>
                    <td className={styles.apCell}>{(ap * 100).toFixed(1)}%</td>
                    <td className={styles.barCell}>
                      <div className={styles.barBg}>
                        <div
                          className={styles.barFill}
                          style={{ width: `${ap * 100}%`, background: ap > 0.5 ? "#34d399" : ap > 0.2 ? "#f59e0b" : "#f87171" }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
