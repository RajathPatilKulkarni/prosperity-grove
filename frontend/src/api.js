export async function runExperiment(payload) {
  const response = await fetch("http://127.0.0.1:8000/run-experiment", {
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