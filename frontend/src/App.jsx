import { runExperiment } from "./api";

export default function App() {
  const handleRun = async () => {
    try {
      const response = await runExperiment({
        scenario: "bull",
        agent_type: "ppo",
        episodes: 10,
        timesteps: 15000,
      });

      console.log("Experiment result:", response);
      alert("Simulation completed. Check console for results.");
    } catch (err) {
      console.error(err);
      alert("Error running simulation");
    }
  };

  return (
    <div className="app-container">
      <h1>🌳 Prosperity Grove</h1>
      <p className="subtitle">
        Visualizing long-term financial intelligence
      </p>

      <button onClick={handleRun}>
        Run Simulation
      </button>
    </div>
  );
}
