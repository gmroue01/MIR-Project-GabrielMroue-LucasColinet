import React, { useState, useEffect, useCallback } from "react";
import { getImages, getClasses } from "../api";
import styles from "./Step1Photo.module.css";

const PAGE_SIZE = 48;

export default function Step1Photo({ selected, onSelect, onNext }) {
  const [images, setImages]           = useState([]);
  const [classes, setClasses]         = useState([]);
  const [classFilter, setClassFilter] = useState("");
  const [page, setPage]               = useState(1);
  const [total, setTotal]             = useState(0);
  const [loading, setLoading]         = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const d = await getImages(page, PAGE_SIZE, classFilter);
      setImages(d.images);
      setTotal(d.total);
    } catch {
      setImages([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [page, classFilter]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { getClasses().then(d => setClasses(d.classes)).catch(() => {}); }, []);

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const changeFilter = (v) => { setClassFilter(v); setPage(1); };

  return (
    <div className={styles.step}>
      {/* Header */}
      <div className={styles.head}>
        <div className={styles.stepId}>01</div>
        <div>
          <h2 className={styles.title}>Sélectionner une image</h2>
          <p className={styles.sub}>Choisissez l'image requête pour la recherche</p>
        </div>
        <div className={styles.headRight}>
          <select
            className={styles.select}
            value={classFilter}
            onChange={e => changeFilter(e.target.value)}
          >
            <option value="">Toutes les classes ({total})</option>
            {classes.map(c => (
              <option key={c} value={c}>{c.replace(/_/g, " ")}</option>
            ))}
          </select>
          {totalPages > 1 && (
            <div className={styles.pager}>
              <button className={styles.pageBtn} onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>‹</button>
              <span className={styles.pageInfo}>{page} / {totalPages}</span>
              <button className={styles.pageBtn} onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}>›</button>
            </div>
          )}
        </div>
      </div>

      {/* Grid */}
      <div className={styles.body}>
        {loading ? (
          <div className={styles.loading}>
            <span className={styles.loadDot} />
            <span className={styles.loadDot} />
            <span className={styles.loadDot} />
          </div>
        ) : (
          <div className={styles.grid}>
            {images.map(img => (
              <button
                key={img.index}
                className={`${styles.card} ${selected?.index === img.index ? styles.cardOn : ""}`}
                onClick={() => onSelect(img)}
                title={img.class.replace(/_/g, " ")}
              >
                <img src={`/images/${img.filename}`} alt={img.filename} className={styles.thumb} />
                {selected?.index === img.index && <span className={styles.checkmark}>✓</span>}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className={styles.foot}>
        {selected ? (
          <div className={styles.footInner}>
            <div className={styles.selectedInfo}>
              <img src={`/images/${selected.filename}`} alt="selected" className={styles.selectedThumb} />
              <div>
                <span className={styles.selectedLabel}>Image sélectionnée</span>
                <span className={styles.selectedClass}>{selected.class.replace(/_/g, " ")}</span>
              </div>
            </div>
            <button className={styles.nextBtn} onClick={onNext}>
              Continuer <span className={styles.arrow}>→</span>
            </button>
          </div>
        ) : (
          <p className={styles.hint}>Cliquez sur une image pour la sélectionner</p>
        )}
      </div>
    </div>
  );
}
