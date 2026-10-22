const $ = (id) => document.getElementById(id);
const CIRC = 2 * Math.PI * 54;
const COLORS = { human: "#16a34a", uncertain: "#d97706", ai: "#dc2626" };
const NOTES = {
  human: "The writing patterns look more like a human author.",
  uncertain: "The evidence is mixed, so no confident call can be made.",
  ai: "The writing patterns resemble AI-generated text.",
};
const SAMPLES = {
  human: "Honestly, I was not sure what to expect when I started this project. The first week I spent more time fixing my Python installation than writing code, and my laptop kept freezing whenever I opened too many tabs. Once the model finally trained, it felt good even though the results were messy.",
  ai: "Artificial intelligence has transformed numerous industries by enabling organizations to automate complex processes and derive actionable insights from large datasets. Furthermore, responsible deployment requires careful consideration of ethical implications, transparency, and accountability. Overall, a balanced approach ensures sustainable innovation.",
};
const input = $("input"), arc = $("gaugeArc");
arc.setAttribute("stroke-dasharray", CIRC); arc.setAttribute("stroke-dashoffset", CIRC);
let timer = null, ctrl = null;
function countWords(text) { const match = text.trim().match(/\S+/g); return match ? match.length : 0; }
function updateCounts() { $("wordCount").textContent = countWords(input.value); $("charCount").textContent = input.value.length; }
function setGauge(probability, color) { arc.setAttribute("stroke-dashoffset", CIRC * (1 - probability)); arc.style.stroke = color; }
function idle(message) {
  setGauge(0, "#cbd5e1"); $("gaugeValue").textContent = "--";
  $("verdict").textContent = "Waiting"; $("verdict").className = "verdict idle";
  $("verdictNote").textContent = message; $("statWords").textContent = countWords(input.value);
  $("statSents").textContent = "-"; $("statLatency").textContent = "-"; $("sentList").innerHTML = '<p class="empty">Sentence-level analysis will appear here.</p>';
}
function render(data) {
  setGauge(data.p_ai, COLORS[data.verdict]); $("gaugeValue").textContent = `${(data.p_ai * 100).toFixed(0)}%`;
  $("verdict").textContent = data.label; $("verdict").className = `verdict ${data.verdict}`;
  $("verdictNote").textContent = `${NOTES[data.verdict]} (AI probability ${(data.p_ai * 100).toFixed(1)}%)`;
  $("statWords").textContent = data.words; $("statSents").textContent = data.sentence_count; $("statLatency").textContent = data.latency_ms;
  const list = $("sentList"); list.innerHTML = "";
  data.sentences.forEach((item) => {
    const row = document.createElement("div"); row.className = "sent";
    const text = document.createElement("span"); text.textContent = item.sentence;
    const tag = document.createElement("span"); tag.className = "tag";
    if (item.p_ai === null) { row.classList.add("skip"); tag.textContent = "too short"; }
    else { const hue = 120 * (1 - item.p_ai); row.style.background = `hsla(${hue},70%,50%,.15)`; row.style.borderLeftColor = `hsl(${hue},65%,42%)`; tag.textContent = `${(item.p_ai * 100).toFixed(0)}% AI`; }
    row.append(text, tag); list.appendChild(row);
  });
}
async function run() {
  if (ctrl) ctrl.abort(); const text = input.value;
  if (!text.trim()) { $("status").textContent = ""; idle("Type or paste text to begin."); return; }
  ctrl = new AbortController(); $("status").textContent = "Analyzing...";
  try {
    const response = await fetch("/predict", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text}), signal: ctrl.signal });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const data = await response.json(); $("status").textContent = "";
    if (data.status !== "ok") { idle(`Need at least ${data.min_words} words for a reliable result (you have ${data.words}).`); return; }
    render(data);
  } catch (error) { if (error.name !== "AbortError") $("status").textContent = `Error: ${error.message}`; }
}
function schedule() { updateCounts(); clearTimeout(timer); timer = setTimeout(run, 450); }
input.addEventListener("input", schedule);
$("btnHuman").addEventListener("click", () => { input.value = SAMPLES.human; schedule(); });
$("btnAI").addEventListener("click", () => { input.value = SAMPLES.ai; schedule(); });
$("btnClear").addEventListener("click", () => { input.value = ""; schedule(); });
$("fileInput").addEventListener("change", (event) => { const file = event.target.files[0]; if (!file) return; const reader = new FileReader(); reader.onload = () => { input.value = String(reader.result).slice(0, 30000); schedule(); }; reader.readAsText(file); event.target.value = ""; });
async function loadMetrics() {
  try {
    const metrics = await (await fetch("/metrics")).json();
    if (metrics.accuracy === undefined) { $("metricsNote").textContent = "Train the model to see evaluation metrics here."; return; }
    const percent = (value) => `${(value * 100).toFixed(1)}%`;
    [["Accuracy", percent(metrics.accuracy)], ["Precision", percent(metrics.precision)], ["Recall", percent(metrics.recall)], ["F1-score", percent(metrics.f1)], ["ROC-AUC", metrics.auc.toFixed(3)], ["False-positive rate", percent(metrics.false_positive_rate)]].forEach(([key, value]) => {
      const card = document.createElement("div"); card.className = "metric"; card.innerHTML = `<div class="metric-val">${value}</div><div class="metric-key">${key}</div>`; $("metricsGrid").append(card);
    });
    const matrix = metrics.confusion_matrix; $("cmTN").textContent = matrix.tn; $("cmFP").textContent = matrix.fp; $("cmFN").textContent = matrix.fn; $("cmTP").textContent = matrix.tp; $("cmCard").hidden = false;
    $("metricsNote").textContent = `Evaluated on ${metrics.test_samples} held-out texts (trained on ${metrics.train_samples}).`;
  } catch { $("metricsNote").textContent = "Metrics are unavailable right now."; }
}
idle("Type or paste text to begin."); loadMetrics();
