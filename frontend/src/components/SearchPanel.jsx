import React, { useState } from "react";
import { search } from "../api";
import styles from "./SearchPanel.module.css";

const DESCRIPTORS = [
  { id: "color_histogram",    label: "Histo. Couleur" },
  { id: "mobilenet_arcface",  label: "MobileNet ArcFace" },
  { id: "mobilenet_zeroshot", label: "MobileNet ZeroShot" },
  { id: "resnet50_zeroshot",  label: "ResNet50 ZeroShot" },
  { id: "vit_b16_zeroshot",   label: "ViT-B/16 ZeroShot" },
  { id: "dinov2_supcon",      label: "DinoV2 SupCon" },
  { id: "dinov2_zeroshot",    label: "DinoV2 ZeroShot" },
];

const MEASURES = [
  { id: "euclidean", label: "Euclidienne" },
  { id: "cosine",    label: "Cosinus" },
];

export default function SearchPanel({ queryImage, onSearch, setSearching, searching }) {
  const [selectedDescriptors, setSelectedDescriptors] = useState(["color_histogram"]);
  const [measure, setMeasure] = useState("euclidean");
  const [topK, setTopK] = useState(50);
  const [error, setError] = useState("");

  const toggleDescriptor = (id) => {
    setSelectedDescriptors((prev) => {
      const next = prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id];
      return next.length === 0 ? prev : next;
    });
  };

  const handleSearch = async () => {
    if (!queryImage) {
      setError("Veuillez sélectionner une image requête.");
      return;
    }
    setError("");
    setSearching(true);
    try {
      const result = await search(queryImage.index, selectedDescriptors, measure, topK);
      onSearch(result);
    } catch (e) {
      setError(e.response?.data?.detail || "Erreur lors de la recherche.");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className={styles.panel}>
      <h2 className={styles.title}>Paramètres</h2>

      <div className={styles.section}>
        <label className={styles.label}>Image requête</label>
        {queryImage ? (
          <div className={styles.queryPreview}>
            <img src={`/images/${queryImage.filename}`} alt="query" className={styles.queryThumb} />
            <span className={styles.queryName}>{queryImage.class.replace(/_/g, " ")}</span>
          </div>
        ) : (
          <p className={styles.hint}>Sélectionnez une image dans la grille →</p>
        )}
      </div>

      <div className={styles.section}>
        <label className={styles.label}>Descripteurs</label>
        <div className={styles.chips}>
          {DESCRIPTORS.map((d) => (
            <button
              key={d.id}
              className={`${styles.chip} ${selectedDescriptors.includes(d.id) ? styles.chipActive : ""} ${d.soon ? styles.chipSoon : ""}`}
              onClick={() => !d.soon && toggleDescriptor(d.id)}
              disabled={d.soon}
              title={d.soon ? "Bientôt disponible" : ""}
            >
              {d.label}{d.soon && <span className={styles.soonBadge}> bientôt</span>}
            </button>
          ))}
        </div>
        {selectedDescriptors.length > 1 && (
          <p className={styles.hint}>Concaténation : {selectedDescriptors.length} descripteurs</p>
        )}
      </div>

      <div className={styles.section}>
        <label className={styles.label}>Mesure de similarité</label>
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

      <div className={styles.section}>
        <label className={styles.label}>Résultats</label>
        <div className={styles.topkRow}>
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

      <button
        className={styles.searchBtn}
        onClick={handleSearch}
        disabled={searching || !queryImage}
      >
        {searching ? "Recherche en cours..." : "Lancer la recherche"}
      </button>
    </div>
  );
}
