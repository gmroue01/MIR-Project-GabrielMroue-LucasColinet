import React, { useState } from "react";
import styles from "./App.module.css";
import Step1Photo from "./components/Step1Photo";
import Step2Params from "./components/Step2Params";
import Step3Results from "./components/Step3Results";
import Step4Metrics from "./components/Step4Metrics";
import BenchmarkPage from "./components/BenchmarkPage";
import CLIPPage from "./components/CLIPPage";
import LoginPage from "./components/LoginPage";

const STEPS = [
  { n: 1, label: "Photo" },
  { n: 2, label: "Paramètres" },
  { n: 3, label: "Résultats" },
  { n: 4, label: "Métriques" },
];

export default function App() {
  // All hooks must be called unconditionally before any conditional return
  const [authed,     setAuthed]     = useState(() => !!localStorage.getItem("mir_token"));
  const [page,       setPage]       = useState("search");
  const [step,       setStep]       = useState(1);
  const [queryImage, setQueryImage] = useState(null);
  const [config,     setConfig]     = useState({ descriptors: ["color_histogram"], measure: "euclidean", topK: 50 });
  const [result,     setResult]     = useState(null);
  const [searching,  setSearching]  = useState(false);
  const [searchId,   setSearchId]   = useState(0);

  if (!authed) return <LoginPage onLogin={() => setAuthed(true)} />;

  const accessible = (n) => n <= 2 || !!result;
  const go = (n) => { if (accessible(n)) setStep(n); };

  const handleSearchDone = (r) => { setResult(r); setSearchId(id => id + 1); go(3); };
  const handleNewSearch  = () => { setResult(null); setStep(1); };

  const handleRerank = async (poolPercent) => {
    if (!queryImage) return;
    setSearching(true);
    try {
      const r = await (await import("./api")).search(
        queryImage.index, config.descriptors, config.measure, config.topK,
        true, poolPercent,
      );
      setResult(r);
    } catch (e) {
      console.error(e);
    } finally {
      setSearching(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("mir_token");
    setAuthed(false);
  };

  return (
    <div className={styles.app}>
      <div className={styles.stars} aria-hidden="true" />

      <nav className={styles.nav}>
        <span className={styles.logo}><span className={styles.star}>✦</span> MIR</span>
        <div className={styles.navLinks}>
          <button className={`${styles.navBtn} ${page === "search"    ? styles.navOn : ""}`} onClick={() => setPage("search")}>Recherche</button>
          <button className={`${styles.navBtn} ${page === "benchmark" ? styles.navOn : ""}`} onClick={() => setPage("benchmark")}>Benchmark</button>
          <button className={`${styles.navBtn} ${page === "clip"      ? styles.navOn : ""}`} onClick={() => setPage("clip")}>CLIP</button>
          <button className={styles.navBtn} onClick={handleLogout} title="Déconnexion" style={{ color: "#475569" }}>⏻</button>
        </div>
      </nav>

      {page === "clip" ? (
        <div className={styles.benchWrap}><CLIPPage /></div>
      ) : page === "benchmark" ? (
        <div className={styles.benchWrap}><BenchmarkPage /></div>
      ) : (
        <>
          <aside className={styles.stepNav}>
            {STEPS.map((s, i) => (
              <React.Fragment key={s.n}>
                <button
                  className={[styles.dot, step === s.n && styles.dotOn, !accessible(s.n) && styles.dotOff].filter(Boolean).join(" ")}
                  onClick={() => go(s.n)}
                  title={s.label}
                  disabled={!accessible(s.n)}
                >
                  <span className={styles.dotRing} />
                  <span className={styles.dotCore} />
                  <span className={styles.dotLbl}>{s.label}</span>
                </button>
                {i < 3 && <div className={`${styles.line} ${step > s.n ? styles.lineDone : ""}`} />}
              </React.Fragment>
            ))}
          </aside>

          <div className={styles.viewport}>
            <div
              className={styles.slider}
              style={{ transform: `translateY(calc(${-(step - 1)} * (100vh - 56px)))` }}
            >
              <section className={styles.slide}>
                <Step1Photo selected={queryImage} onSelect={setQueryImage} onNext={() => go(2)} />
              </section>
              <section className={styles.slide}>
                <Step2Params
                  queryImage={queryImage}
                  config={config}
                  onChange={setConfig}
                  onSearch={handleSearchDone}
                  searching={searching}
                  setSearching={setSearching}
                  onBack={() => go(1)}
                />
              </section>
              <section className={styles.slide}>
                <Step3Results
                  result={result}
                  onNext={() => go(4)}
                  onNewSearch={handleNewSearch}
                  onRerank={handleRerank}
                  reranking={searching}
                  topK={config.topK}
                />
              </section>
              <section className={styles.slide}>
                <Step4Metrics key={searchId} result={result} config={config} onNewSearch={handleNewSearch} />
              </section>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
