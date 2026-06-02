import React, { useState, useEffect, useCallback } from "react";
import { getImages, getClasses } from "../api";
import styles from "./ImageBrowser.module.css";

export default function ImageBrowser({ onSelect, selectedImage }) {
  const [images, setImages] = useState([]);
  const [classes, setClasses] = useState([]);
  const [classFilter, setClassFilter] = useState("");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const PAGE_SIZE = 48;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getImages(page, PAGE_SIZE, classFilter);
      setImages(data.images);
      setTotal(data.total);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [page, classFilter]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    getClasses().then((d) => setClasses(d.classes)).catch(() => {});
  }, []);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className={styles.browser}>
      <div className={styles.controls}>
        <div className={styles.controlRow}>
          <select
            value={classFilter}
            onChange={(e) => { setClassFilter(e.target.value); setPage(1); }}
            className={styles.select}
          >
            <option value="">Toutes les classes ({total})</option>
            {classes.map((c) => (
              <option key={c} value={c}>{c.replace(/_/g, " ")}</option>
            ))}
          </select>
        </div>
        {selectedImage && (
          <div className={styles.selectedBadge}>
            Image sélectionnée : <strong>{selectedImage.filename}</strong>
          </div>
        )}
      </div>

      {loading ? (
        <div className={styles.loading}>Chargement...</div>
      ) : (
        <div className={styles.grid}>
          {images.map((img) => (
            <button
              key={img.index}
              className={`${styles.card} ${selectedImage?.index === img.index ? styles.selected : ""}`}
              onClick={() => onSelect(img)}
              title={img.class.replace(/_/g, " ")}
            >
              <img
                src={`/images/${img.filename}`}
                alt={img.filename}
                className={styles.thumb}
              />
            </button>
          ))}
        </div>
      )}

      {totalPages > 1 && (
        <div className={styles.pagination}>
          <button onClick={() => setPage(1)} disabled={page === 1} className={styles.pageBtn}>«</button>
          <button onClick={() => setPage(p => p - 1)} disabled={page === 1} className={styles.pageBtn}>‹</button>
          <span className={styles.pageInfo}>{page} / {totalPages}</span>
          <button onClick={() => setPage(p => p + 1)} disabled={page === totalPages} className={styles.pageBtn}>›</button>
          <button onClick={() => setPage(totalPages)} disabled={page === totalPages} className={styles.pageBtn}>»</button>
        </div>
      )}
    </div>
  );
}
