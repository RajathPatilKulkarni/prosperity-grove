const LOCAL_API_BASE = "http://127.0.0.1:8000";
const DEPLOYED_API_BASE = "https://prosperity-grove.onrender.com";

const isLocalHost = ["localhost", "127.0.0.1", ""].includes(
  window.location.hostname
);

const API_BASE = (
  import.meta.env.VITE_API_BASE ||
  (isLocalHost ? LOCAL_API_BASE : DEPLOYED_API_BASE)
).replace(/\/$/, "");

const readErrorMessage = async (response) => {
  const text = await response.text();
  if (!text) return "Request failed";

  try {
    const payload = JSON.parse(text);
    return payload.detail || payload.message || text;
  } catch {
    return text;
  }
};

export async function runExperiment(payload) {
  const response = await fetch(`${API_BASE}/run-experiment`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

export async function runExperimentStream(payload, onEvent) {
  const response = await fetch(`${API_BASE}/run-experiment/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(
      response.ok ? "Stream failed" : await readErrorMessage(response)
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line);
        onEvent?.(event);
      } catch {
        onEvent?.({
          type: "error",
          message: "Stream parse error",
        });
      }
    }
  }
}
