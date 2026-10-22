import { useEffect, useRef, useState } from "react";
import { signIn, signUp, signOutUser, isFirebaseConfigured } from "./firebase";
import { getMetrics, predict, humanize } from "./api";

const countWords = (value) => (value.trim().match(/\S+/g) || []).length;
const SAMPLE_TEXT = `A clear workflow helps teams make better decisions. Start by defining the problem, then collect the evidence needed to understand it. Review the results with the people who use the process every day, because practical feedback often reveals details that a report misses. Finally, document the decision and measure the outcome so the next iteration can be more focused.`;

function Metric({ value, label }) {
  return <div className="metric"><strong>{value}</strong><span>{label}</span></div>;
}

function AuthModal({ onClose, onAuthenticated }) {
  const [mode, setMode] = useState("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const user = mode === "signin" ? await signIn(email, password) : await signUp(email, password);
      onAuthenticated(user);
      onClose();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="auth-card" role="dialog" aria-modal="true" aria-labelledby="auth-title">
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        <div className="brand large"><span className="brand-mark">✦</span><span>Detect<span className="brand-accent">AI</span></span></div>
        <h2 id="auth-title">{mode === "signin" ? "Welcome back" : "Create your account"}</h2>
        <p className="muted">Save your analysis workspace and access it anywhere.</p>
        {!isFirebaseConfigured && <div className="notice">Add the Firebase VITE_FIREBASE_* values to frontend/.env.local before signing in.</div>}
        <form onSubmit={submit}>
          <label>Email<input type="email" required value={email} onChange={(event) => setEmail(event.target.value)} /></label>
          <label>Password<input type="password" minLength="6" required value={password} onChange={(event) => setPassword(event.target.value)} /></label>
          {error && <p className="form-error">{error}</p>}
          <button className="button primary full" disabled={busy || !isFirebaseConfigured}>{busy ? "Please wait..." : mode === "signin" ? "Sign in" : "Create account"}</button>
        </form>
        <button className="switch-auth" onClick={() => { setMode(mode === "signin" ? "signup" : "signin"); setError(""); }}>
          {mode === "signin" ? "New to DetectAI? Create an account" : "Already have an account? Sign in"}
        </button>
      </div>
    </div>
  );
}

function Result({ result }) {
  if (!result) return <div className="empty-result"><span className="result-icon">✦</span><h3>Your result will appear here</h3><p>Upload a document or paste your text, then press Analyze.</p></div>;
  const percent = Math.round(result.p_ai * 100);
  const confidence = Math.round(Math.abs(result.p_ai - 0.5) * 200);
  const flaggedSentences = (result.sentences || []).filter((item) => item.p_ai !== null);
  return <div className="result-content">
    <div className="result-heading"><span>Analysis complete</span><span className="confidence-dot" /><span>{confidence}% confidence</span></div>
    <div className={`score score-${result.verdict}`}><strong>{percent}%</strong><span>AI probability</span></div>
    <div className={`verdict verdict-${result.verdict}`}>{result.label}</div>
    <p className="result-summary">{result.verdict === "ai" ? "The language patterns strongly resemble AI-generated writing." : result.verdict === "human" ? "The language patterns are more consistent with human writing." : "The language signals are mixed. Treat this result as inconclusive."}</p>
    <div className="result-bar"><span style={{ width: `${percent}%` }} /></div>
    <div className="result-stats"><span><b>{result.words}</b> words</span><span><b>{result.sentence_count}</b> sentences</span><span><b>{result.latency_ms}</b> ms</span></div>
    {flaggedSentences.length > 0 && <details className="sentence-details"><summary>View sentence signals</summary><div className="sentence-list">{flaggedSentences.map((item, index) => <div className="sentence-row" key={`${item.sentence}-${index}`}><span>{item.sentence}</span><b>{Math.round(item.p_ai * 100)}%</b></div>)}</div></details>}
    <p className="disclaimer">This is an ML estimate, not proof of authorship.</p>
  </div>;
}

export default function App() {
  const [user, setUser] = useState(null);
  const [authOpen, setAuthOpen] = useState(false);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [humanized, setHumanized] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef(null);

  useEffect(() => {
    getMetrics().then(setMetrics).catch((error) => setMetricsError(error.message));
  }, []);

  async function analyze() {
    if (!text.trim()) {
      setMessage("Paste text or upload a document first.");
      return;
    }
    setBusy(true);
    setMessage("");
    setHumanized("");
    try {
      const data = await predict(text);
      if (data.status === "ok") setResult(data);
      else setMessage(`Please enter at least ${data.min_words} words. You currently have ${data.words}.`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function convertToHuman() {
    setBusy(true);
    setMessage("");
    try {
      setHumanized(await humanize(text));
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  function readFile(file) {
    if (!file) return;
    const allowed = /\.(txt|md|csv)$/i.test(file.name);
    if (!allowed) {
      setMessage("Please upload a .txt, .md, or .csv file.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => { setText(String(reader.result).slice(0, 30000)); setResult(null); setHumanized(""); setMessage(""); };
    reader.onerror = () => setMessage("We could not read that file. Please try again.");
    reader.readAsText(file);
  }

  function upload(event) {
    readFile(event.target.files[0]);
    event.target.value = "";
  }

  return <div className="app-shell">
    <header className="dashboard-nav"><a className="brand" href="#top"><span className="brand-mark">✦</span><span>Detect<span className="brand-accent">AI</span></span></a><div className="nav-right">{user ? <><span className="user-email">{user.email}</span><button className="button nav-button" onClick={() => { signOutUser(); setUser(null); }}>Sign out</button></> : <button className="button nav-button" onClick={() => setAuthOpen(true)}>Sign in</button>}</div></header>
    <main className="dashboard" id="top">
      <div className="dashboard-heading"><div><p className="eyebrow">AI writing intelligence</p><h1>Understand your text.</h1><p className="subtitle">A fast, transparent analysis powered by supervised language patterns. Get a probability, not a black-box verdict.</p></div><div className="secure-badge">● Private analysis</div></div>
      {metrics && <section className="metrics-strip" aria-label="Model validation metrics"><div><span className="metrics-kicker">Model validation</span><b>Held-out test performance</b></div><Metric value={`${(metrics.accuracy * 100).toFixed(1)}%`} label="accuracy" /><Metric value={`${(metrics.auc * 100).toFixed(1)}%`} label="ROC AUC" /><Metric value={metrics.test_samples.toLocaleString()} label="test samples" /><span className="metrics-note">Results are estimates, not proof.</span></section>}
      {metricsError && <p className="metrics-error">Live validation metrics unavailable: {metricsError}</p>}
      <div className="analysis-layout">
        <section className="workspace-card">
          <div className="workspace-toolbar"><span className="file-label">Your text</span><span className="word-count">{countWords(text)} / 30,000 characters</span></div>
          <textarea value={text} onChange={(event) => { setText(event.target.value); setResult(null); setHumanized(""); }} maxLength={30000} placeholder="Paste at least 30 words for the most reliable estimate..." />
          <div className="editor-actions"><button className="text-button" onClick={() => { setText(SAMPLE_TEXT); setResult(null); setMessage(""); }}>Try sample</button>{text && <button className="text-button" onClick={() => { setText(""); setResult(null); setHumanized(""); setMessage(""); }}>Clear</button>}</div>
          <div className="upload-row"><button className={`upload-box ${dragging ? "dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); readFile(event.dataTransfer.files[0]); }} onClick={() => fileRef.current?.click()}><span className="upload-icon">↑</span><span><b>Drop a document here or browse</b><small>.txt, .md, or .csv · up to 30,000 characters</small></span></button><input ref={fileRef} type="file" hidden accept=".txt,.md,.csv,text/plain,text/markdown,text/csv" onChange={upload} /><button className="button analyze-button" onClick={analyze} disabled={busy || !text.trim()}>{busy ? "Analyzing..." : "Analyze text  →"}</button></div>
          {message && <p className="form-error">{message}</p>}
        </section>
        <section className="workspace-card result-card"><Result result={result} />{result && <button className="button humanize-button" onClick={convertToHuman} disabled={busy}>✧ Convert to human style</button>}</section>
      </div>
      {humanized && <section className="workspace-card humanized-card"><div className="workspace-toolbar"><span className="file-label">Human-style draft</span><button className="button copy-button" onClick={() => navigator.clipboard.writeText(humanized)}>Copy text</button></div><textarea value={humanized} readOnly /></section>}
      <div className="dashboard-foot"><span>DetectAI uses a supervised NLP model trained on balanced human/AI text.</span><span>Use results responsibly — never as sole evidence.</span></div>
    </main>
    {authOpen && <AuthModal onClose={() => setAuthOpen(false)} onAuthenticated={setUser} />}
  </div>;
}
