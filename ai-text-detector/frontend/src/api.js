export async function predict(text, signal) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Server returned ${response.status}`);
  return data;
}

export async function getMetrics() {
  const response = await fetch("/metrics");
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Server returned ${response.status}`);
  return data;
}

export async function humanize(text) {
  const response = await fetch("/humanize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `Server returned ${response.status}`);
  return data.text;
}
