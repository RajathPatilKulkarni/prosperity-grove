const API_BASE =
  import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function runExperiment(payload) {
  const response = await fetch(`${API_BASE}/run-experiment`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(error);
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
    const error = await response.text();
    throw new Error(error || "Stream failed");
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
      } catch (error) {
        onEvent?.({
          type: "error",
          message: "Stream parse error",
        });
      }
    }
  }
}
