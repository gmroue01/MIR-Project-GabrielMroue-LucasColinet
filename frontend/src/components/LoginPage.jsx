import React, { useState } from "react";
import { login } from "../api";
import styles from "./LoginPage.module.css";

export default function LoginPage({ onLogin }) {
  const [password, setPassword] = useState("");
  const [show,     setShow]     = useState(false);
  const [error,    setError]    = useState("");
  const [loading,  setLoading]  = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!password.trim()) return;
    setError("");
    setLoading(true);
    try {
      const { token } = await login(password);
      localStorage.setItem("mir_token", token);
      onLogin();
    } catch {
      setError("Mot de passe incorrect");
      setPassword("");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.stars} aria-hidden="true" />

      <div className={styles.card}>
        <div className={styles.logo}>
          <span className={styles.star}>✦</span>
          <span className={styles.logoText}>MIR</span>
        </div>

        <h1 className={styles.title}>Moteur de Recherche d'Images</h1>
        <p className={styles.sub}>Entrez le mot de passe pour accéder</p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.inputWrap}>
            <input
              type={show ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Mot de passe"
              className={`${styles.input} ${error ? styles.inputError : ""}`}
              autoFocus
              autoComplete="current-password"
            />
            <button
              type="button"
              className={styles.toggleBtn}
              onClick={() => setShow(s => !s)}
              tabIndex={-1}
            >
              {show ? "○" : "●"}
            </button>
          </div>

          {error && <p className={styles.error}>{error}</p>}

          <button
            type="submit"
            className={styles.submit}
            disabled={loading || !password.trim()}
          >
            {loading ? <span className={styles.spinner} /> : "Connexion →"}
          </button>
        </form>
      </div>
    </div>
  );
}
