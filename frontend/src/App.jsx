import { useState, useEffect } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

const EXAMPLES = [
  "What is retrieval-augmented generation?",
  "How does dense passage retrieval work?",
  "What is a reward function in reinforcement learning?",
  "What are scaling laws for large language models?",
];

function IconLogo() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l1.9 4.8L18.7 9.7 14 11.6 12 16.4 10 11.6 5.3 9.7l4.8-1.9z" />
    </svg>
  );
}
function IconSearch() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}
function IconSend() {
  return (
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="19" x2="12" y2="5" /><polyline points="5 12 12 5 19 12" />
    </svg>
  );
}
function IconMessage() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 11.5a8.5 8.5 0 0 1-12.5 7.5L3 21l2-5.5A8.5 8.5 0 1 1 21 11.5z" />
    </svg>
  );
}
function IconFile() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><polyline points="14 3 14 8 19 8" />
    </svg>
  );
}

function ConfidenceRing({ value }) {
  const r = 23;
  const C = 2 * Math.PI * r;
  const dash = Math.max(0, Math.min(1, value)) * C;
  return (
    <svg width="58" height="58" viewBox="0 0 58 58">
      <circle cx="29" cy="29" r={r} fill="none" stroke="#EEF0F3" strokeWidth="6" />
      <circle cx="29" cy="29" r={r} fill="none" stroke="#10B981" strokeWidth="6"
        strokeLinecap="round" strokeDasharray={`${dash} ${C}`} transform="rotate(-90 29 29)" />
      <text x="29" y="29" textAnchor="middle" dominantBaseline="central" fontSize="14" fontWeight="600" fill="#0F172A">
        {Math.round(value * 100)}%
      </text>
    </svg>
  );
}

function renderAnswer(text) {
  return text.split(/(\[\d+\])/g).map((p, i) => {
    const m = p.match(/^\[(\d+)\]$/);
    return m ? <span className="cite" key={i}>{m[1]}</span> : <span key={i}>{p}</span>;
  });
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/health`).then((r) => r.json()).then(setHealth).catch(() => setHealth(null));
  }, []);

  async function ask(q) {
    const query = q ?? question;
    if (!query.trim()) return;
    setQuestion(query);
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const maxScore = result?.sources?.length ? Math.max(...result.sources.map((s) => s.score)) : 1;

  return (
    <div className="page">
      <div className="app">
        <header className="header">
          <div className="brand">
            <div className="logo"><IconLogo /></div>
            <div>
              <div className="brand-name">Document Intelligence</div>
              <div className="brand-sub">Ask anything across your documents</div>
            </div>
          </div>
          <div className="status">
            <span className="dot" />
            {health
              ? health.llm_provider === "groq" ? "Connected · Groq" : `Connected · ${health.llm_provider}`
              : "Connecting…"}
          </div>
        </header>

        <div className="searchbar">
          <span className="search-ico"><IconSearch /></span>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="Ask a question about your documents…"
          />
          <button className="send" onClick={() => ask()} disabled={loading} aria-label="Ask">
            <IconSend />
          </button>
        </div>

        <div className="chips">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="chip" onClick={() => ask(ex)}>{ex}</button>
          ))}
        </div>

        {loading && <div className="hint">Thinking…</div>}
        {error && <div className="error">Error: {error}</div>}

        {result && (
          <>
            <div className="answer-card">
              <div className="avatar"><IconMessage /></div>
              <div className="answer-text">{renderAnswer(result.answer)}</div>
            </div>

            <div className="panels">
              <div className="panel conf-panel">
                <ConfidenceRing value={result.router_confidence} />
                <div>
                  <div className="panel-label">Router confidence</div>
                  <div className="panel-big">Category: {result.category}</div>
                  <div className="panel-note">XGBoost classifier</div>
                </div>
              </div>
              <div className="panel time-panel">
                <div className="panel-label">Response time</div>
                <div className="time-val">{Math.round(result.latency_ms)}<span className="unit"> ms</span></div>
              </div>
            </div>

            <div className="sources-label">Sources · ranked by relevance</div>
            {result.sources.map((s, i) => {
              const pct = Math.round((s.score / maxScore) * 100);
              return (
                <div className="source" key={s.chunk_id}>
                  <div className="source-head">
                    <span className={`doc ${i === 0 ? "doc-top" : ""}`}>
                      <span className="file-ico"><IconFile /></span> {s.doc_id}
                    </span>
                    <span className={`match ${i === 0 ? "match-top" : ""}`}>{pct}% match</span>
                  </div>
                  <div className="bar-track">
                    <div className={`bar-fill ${i === 0 ? "top" : "rest"}`} style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}