import { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

function percent(value) {
  return typeof value === "number" ? `${value.toFixed(2)}%` : "—";
}

function number(value) {
  return typeof value === "number" ? value.toFixed(4) : "—";
}

function Header() {
  return <header><div><strong>RemoteSensing-VLM</strong><span>BigEarthNet baseline scene analysis</span></div><span className="status">● API ready</span></header>;
}

function FileInput({ label, description, file, onChange }) {
  return <label className="upload-card"><span className="satellite">{label}</span><b>{label === "S1" ? "Sentinel-1 radar" : "Sentinel-2 optical"}</b><small>{description}</small><input type="file" accept=".zip" onChange={(event) => onChange(event.target.files?.[0] ?? null)} /><em>{file ? file.name : "Choose ZIP archive"}</em></label>;
}

function Upload({ onResult }) {
  const [s1, setS1] = useState(null);
  const [s2, setS2] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    if (!s1 || !s2) return setError("Select one Sentinel-1 ZIP and one matching Sentinel-2 ZIP.");
    setLoading(true); setError("");
    const form = new FormData(); form.append("s1_zip", s1); form.append("s2_zip", s2);
    try {
      const response = await fetch(`${API_URL}/predict/upload`, { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Analysis failed.");
      onResult(body);
    } catch (err) { setError(err.message || "Unable to reach the API."); }
    finally { setLoading(false); }
  }

  return <main className="shell landing"><p className="eyebrow">MULTIMODAL EARTH OBSERVATION</p><h1>Land-cover analysis from paired Sentinel scenes.</h1><p className="lead">This baseline combines a pretrained 12-channel BigEarthNet ResNet-50 prediction with transparent spectral and radar observations. It is not a Transformer or VLM.</p><form onSubmit={submit}><div className="uploads"><FileInput label="S1" description="VV and VH radar bands" file={s1} onChange={setS1} /><FileInput label="S2" description="B02–B12 multispectral bands" file={s2} onChange={setS2} /></div>{error && <p className="error">{error}</p>}<button disabled={loading}>{loading ? "Analysing scene…" : "Run scene analysis"}</button></form><p className="hint">Each archive must contain exactly one matching patch. Required bands and finite raster values are checked before inference.</p></main>;
}

function Metric({ label, value, note }) { return <article className="metric"><small>{label}</small><strong>{value}</strong>{note && <span>{note}</span>}</article>; }

function Results({ result, onReset }) {
  const { prediction, spectral_evidence: spectral, radar_evidence: radar, category_evidence: categories, evidence_confidence: confidence, explanation, methodology } = result;
  const top = prediction.top_prediction;
  return <main className="shell results"><div className="result-heading"><div><p className="eyebrow">BASELINE RESULT</p><h1>{top.label}</h1><p>Highest-scoring label from the 19-class multi-label model.</p></div><button className="secondary" onClick={onReset}>Analyse another pair</button></div><section className="metrics"><Metric label="Top-label sigmoid output" value={percent(top.confidence_percent)} note="Not calibrated confidence" /><Metric label="Heuristic consistency" value={percent(confidence.heuristic_evidence_score)} note={confidence.consistency_level.replaceAll("_", " ")} /><Metric label="Detected labels" value={prediction.detected_labels.length} note={`Threshold: ${prediction.threshold}`} /><Metric label="Model" value="ResNet-50" note="12 channels · 120 × 120" /></section><section><h2>Model predictions</h2><div className="prediction-list">{prediction.predictions.map((item) => <div key={item.label}><span>{item.label}</span><strong>{percent(item.confidence_percent)}</strong><i className={item.detected ? "detected" : ""}>{item.detected ? "Detected" : "Below threshold"}</i></div>)}</div></section><section><h2>Measured evidence</h2><div className="metrics"><Metric label="NDVI" value={number(spectral.indices.NDVI.mean)} note="Vegetation index" /><Metric label="NDWI" value={number(spectral.indices.NDWI.mean)} note="Water index" /><Metric label="NDBI" value={number(spectral.indices.NDBI.mean)} note="Built-up index" /><Metric label="VV mean" value={number(radar.VV.mean)} note={radar.VV.signal.replaceAll("_", " ")} /><Metric label="VH mean" value={number(radar.VH.mean)} note={radar.VH.signal.replaceAll("_", " ")} /><Metric label="Radar character" value={radar.radar_character.replaceAll("_", " ")} note="Heuristic interpretation" /></div></section><section><h2>Category consistency</h2><div className="categories">{Object.entries(categories).map(([name, item]) => <article key={name}><strong>{name.replaceAll("_", " ")}</strong><span className="tag">{item.evidence_status.replaceAll("_", " ")}</span><p>{item.reason}</p></article>)}</div></section><section className="explanation"><h2>Rule-based explanation</h2><p>{explanation.summary}</p><small>{explanation.note}</small></section><section className="method"><h2>Method and limitations</h2><p>{methodology.note}</p><p>{confidence.note}</p></section></main>;
}

export default function App() {
  const [result, setResult] = useState(null);
  return <><Header />{result ? <Results result={result} onReset={() => setResult(null)} /> : <Upload onResult={setResult} />}</>;
}
