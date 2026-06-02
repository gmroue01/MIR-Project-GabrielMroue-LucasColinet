import React, { useState, useEffect } from "react";
import { getClipImages, clipTextToImage, clipImageToText, clipEvaluate } from "../api";
import styles from "./CLIPPage.module.css";

const PAGE_SIZE = 32;

// ── Shared: Flickr8K image browser ────────────────────────────────────────

function ClipImageBrowser({ onSelect, isSelected }) {
  const [images, setImages] = useState([]);
  const [page, setPage]     = useState(1);
  const [total, setTotal]   = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    getClipImages(page, PAGE_SIZE)
      .then(d => { setImages(d.images); setTotal(d.total); })
      .catch(() => setImages([]))
      .finally(() => setLoading(false));
  }, [page]);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className={styles.browser}>
      {loading ? (
        <div className={styles.dots}><span /><span /><span /></div>
      ) : images.length === 0 ? (
        <p className={styles.empty}>Aucune image disponible — backend non connecté.</p>
      ) : (
        <div className={styles.browserGrid}>
          {images.map(img => (
            <button
              key={img.index}
              className={`${styles.browserThumb} ${isSelected(img) ? styles.thumbOn : ""}`}
              onClick={() => onSelect(img)}
              title={img.filename}
            >
              <img src={`/flickr8k/${img.filename}`} alt="" loading="lazy" />
              {isSelected(img) && <div className={styles.thumbCheck}>✓</div>}
            </button>
          ))}
        </div>
      )}
      {totalPages > 1 && (
        <div className={styles.pager}>
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1 || loading}>‹</button>
          <span>{page} / {totalPages}</span>
          <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages || loading}>›</button>
        </div>
      )}
    </div>
  );
}

// ── Section 01: Text → Image ───────────────────────────────────────────────

function TextToImage() {
  const [query, setQuery]     = useState("");
  const [topK, setTopK]       = useState(10);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const run = async () => {
    if (!query.trim() || loading) return;
    setLoading(true); setError(null);
    try {
      const d = await clipTextToImage(query, topK);
      setResults(d.results);
    } catch (e) {
      setError(e.response?.data?.detail || "Erreur de connexion au backend.");
      setResults([]);
    } finally { setLoading(false); }
  };

  return (
    <section className={styles.section}>
      <div className={styles.sHead}>
        <span className={styles.sNum}>01</span>
        <div>
          <h2 className={styles.sTitle}>Texte → Image</h2>
          <p className={styles.sSub}>Trouvez les images les plus similaires à une description en langage naturel</p>
        </div>
      </div>

      <div className={styles.queryRow}>
        <input
          className={styles.queryInput}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === "Enter" && run()}
          placeholder="Ex: a dog running on the beach at sunset..."
        />
        <TopKPicker value={topK} onChange={setTopK} />
        <RunButton onClick={run} loading={loading} disabled={!query.trim()}>
          Rechercher
        </RunButton>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      {results.length > 0 && (
        <div className={styles.imgGrid}>
          {results.map(r => (
            <div key={r.rank} className={styles.imgCard}>
              <span className={styles.rankBadge}>#{r.rank}</span>
              <img src={`/flickr8k/${r.filename}`} alt="" loading="lazy" className={styles.imgCardImg} />
              <div className={styles.imgCardFoot}>
                <span className={styles.simScore}>{(r.score * 100).toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && !results.length && !error && (
        <p className={styles.hint}>Entrez une description et appuyez sur Entrée ou cliquez Rechercher.</p>
      )}
    </section>
  );
}

// ── Section 02: Image → Text ───────────────────────────────────────────────

function ImageToText() {
  const [selected, setSelected] = useState(null);
  const [topK, setTopK]         = useState(10);
  const [results, setResults]   = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  const run = async () => {
    if (!selected || loading) return;
    setLoading(true); setError(null);
    try {
      const d = await clipImageToText(selected.index, topK);
      setResults(d.results);
    } catch (e) {
      setError(e.response?.data?.detail || "Erreur de connexion au backend.");
      setResults([]);
    } finally { setLoading(false); }
  };

  return (
    <section className={styles.section}>
      <div className={styles.sHead}>
        <span className={styles.sNum}>02</span>
        <div>
          <h2 className={styles.sTitle}>Image → Texte <span className={styles.tag}>Inverse Search</span></h2>
          <p className={styles.sSub}>Retrouvez les descriptions les plus proches d'une image requête</p>
        </div>
      </div>

      <div className={styles.inverseWrap}>
        <div className={styles.invLeft}>
          <p className={styles.panelLbl}>Parcourir Flickr8K</p>
          <ClipImageBrowser
            onSelect={setSelected}
            isSelected={img => img.index === selected?.index}
          />
        </div>

        <div className={styles.invRight}>
          {selected ? (
            <>
              <div className={styles.queryImgWrap}>
                <img src={`/flickr8k/${selected.filename}`} alt="" className={styles.queryImg} />
                <p className={styles.queryImgName}>{selected.filename}</p>
              </div>
              <div className={styles.controlRow}>
                <TopKPicker value={topK} onChange={setTopK} />
                <RunButton onClick={run} loading={loading}>Rechercher</RunButton>
              </div>

              {error && <div className={styles.error}>{error}</div>}

              <div className={styles.captionList}>
                {results.map(r => (
                  <div key={r.rank} className={styles.captionRow}>
                    <span className={styles.captionRank}>#{r.rank}</span>
                    <p className={styles.captionText}>{r.caption}</p>
                    <span className={styles.captionSim}>{(r.score * 100).toFixed(1)}%</span>
                  </div>
                ))}
                {!loading && !results.length && !error && (
                  <p className={styles.hint}>Cliquez "Rechercher" pour trouver les descriptions correspondantes.</p>
                )}
              </div>
            </>
          ) : (
            <div className={styles.placeholder}>
              <span className={styles.placeholderArrow}>←</span>
              <p>Sélectionnez une image dans le navigateur</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

// ── Section 03: Evaluation ─────────────────────────────────────────────────

function Evaluation() {
  const [corpusImgs, setCorpusImgs] = useState([]);
  const [corpusTxts, setCorpusTxts] = useState(["", "", ""]);
  const [topK, setTopK]             = useState(10);
  const [evaluating, setEvaluating] = useState(false);
  const [results, setResults]       = useState(null);
  const [error, setError]           = useState(null);

  const toggleImg = img => {
    setCorpusImgs(prev => {
      const has = prev.find(i => i.index === img.index);
      if (has) return prev.filter(i => i.index !== img.index);
      if (prev.length >= 3) return prev;
      return [...prev, img];
    });
  };

  const run = async () => {
    const texts = corpusTxts.filter(t => t.trim());
    if (!corpusImgs.length && !texts.length) {
      setError("Ajoutez au moins une image ou une description au corpus."); return;
    }
    setEvaluating(true); setError(null);
    try {
      const d = await clipEvaluate(corpusImgs.map(i => i.index), texts, topK);
      setResults(d);
    } catch (e) {
      setError(e.response?.data?.detail || "Erreur de connexion au backend.");
    } finally { setEvaluating(false); }
  };

  const selectedSet = new Set(corpusImgs.map(i => i.index));
  const ready = corpusImgs.length > 0 || corpusTxts.some(t => t.trim());

  return (
    <section className={styles.section}>
      <div className={styles.sHead}>
        <span className={styles.sNum}>03</span>
        <div>
          <h2 className={styles.sTitle}>Évaluation</h2>
          <p className={styles.sSub}>Corpus de 3 images + 3 textes — Précision, Rappel, MAP</p>
        </div>
        <div style={{ marginLeft: "auto" }}>
          <TopKPicker value={topK} onChange={setTopK} />
        </div>
      </div>

      <div className={styles.evalLayout}>
        {/* Left: image corpus */}
        <div className={styles.evalCol}>
          <p className={styles.panelLbl}>
            Corpus Images <span className={styles.badge}>{corpusImgs.length} / 3</span>
          </p>
          <div className={styles.slots}>
            {[0, 1, 2].map(i => (
              <div key={i} className={`${styles.slot} ${corpusImgs[i] ? styles.slotFull : ""}`}>
                {corpusImgs[i] ? (
                  <>
                    <img src={`/flickr8k/${corpusImgs[i].filename}`} alt="" />
                    <button className={styles.rmBtn} onClick={() => toggleImg(corpusImgs[i])}>×</button>
                  </>
                ) : (
                  <span className={styles.slotEmpty}>+</span>
                )}
              </div>
            ))}
          </div>
          <ClipImageBrowser
            onSelect={toggleImg}
            isSelected={img => selectedSet.has(img.index)}
          />
        </div>

        {/* Right: text corpus */}
        <div className={styles.evalCol}>
          <p className={styles.panelLbl}>
            Corpus Textes <span className={styles.badge}>{corpusTxts.filter(t => t.trim()).length} / 3</span>
          </p>
          <p className={styles.evalTip}>
            Idéalement, utilisez des captions Flickr8K pour des métriques de pertinence précises.
          </p>
          {corpusTxts.map((t, i) => (
            <textarea
              key={i}
              className={styles.evalTextarea}
              value={t}
              onChange={e => setCorpusTxts(prev => prev.map((x, j) => j === i ? e.target.value : x))}
              placeholder={`Description ${i + 1} — ex: "a child is jumping into a pool…"`}
              rows={3}
            />
          ))}
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.evalAction}>
        <button className={styles.evalBtn} onClick={run} disabled={evaluating || !ready}>
          {evaluating
            ? <><span className={styles.spin} /> Évaluation en cours…</>
            : "Évaluer →"}
        </button>
      </div>

      {results && <EvalResults results={results} />}
    </section>
  );
}

function EvalResults({ results }) {
  const pct = v => `${(v * 100).toFixed(1)}%`;

  const Block = ({ title, data }) => {
    if (!data) return null;
    return (
      <div className={styles.evalBlock}>
        <h3 className={styles.evalBlockTitle}>{title}</h3>
        <div className={styles.mapDisplay}>
          MAP&nbsp;<strong>{pct(data.map)}</strong>
        </div>
        <table className={styles.evalTable}>
          <thead>
            <tr><th>Requête</th><th>P@k</th><th>R@k</th><th>AP</th></tr>
          </thead>
          <tbody>
            {data.queries?.map((q, i) => (
              <tr key={i}>
                <td className={styles.queryTd}>
                  {q.filename ? (
                    <><img src={`/flickr8k/${q.filename}`} alt="" className={styles.miniThumb} />{q.filename}</>
                  ) : q.text}
                </td>
                <td>{pct(q.precision_at_k)}</td>
                <td>{pct(q.recall_at_k)}</td>
                <td>{pct(q.ap)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className={styles.evalResults}>
      <Block title="Texte → Image" data={results.text_to_image} />
      <Block title="Image → Texte" data={results.image_to_text} />
    </div>
  );
}

// ── Shared small components ────────────────────────────────────────────────

function TopKPicker({ value, onChange }) {
  return (
    <div className={styles.topKRow}>
      {[5, 10, 20].map(k => (
        <button
          key={k}
          className={`${styles.kChip} ${value === k ? styles.kOn : ""}`}
          onClick={() => onChange(k)}
        >
          Top {k}
        </button>
      ))}
    </div>
  );
}

function RunButton({ onClick, loading, disabled = false, children }) {
  return (
    <button className={styles.runBtn} onClick={onClick} disabled={loading || disabled}>
      {loading ? <span className={styles.spin} /> : <>{children} <span>→</span></>}
    </button>
  );
}

// ── Main export ────────────────────────────────────────────────────────────

export default function CLIPPage() {
  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <div className={styles.heroChip}>CLIP · Flickr8K · FAISS · v2.0</div>
        <h1 className={styles.heroTitle}>Recherche Multimodale</h1>
        <p className={styles.heroSub}>
          Embeddings cross-modaux texte / image · Recherche par plus proche voisin avec FAISS
        </p>
      </div>

      <TextToImage />
      <div className={styles.sep} />
      <ImageToText />
      <div className={styles.sep} />
      <Evaluation />
    </div>
  );
}
