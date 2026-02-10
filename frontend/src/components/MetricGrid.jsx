export default function MetricGrid({ items }) {
  return (
    <div className="metric-grid">
      {items.map((item) => (
        <div className="metric-card" key={item.label}>
          <div className="label">{item.label}</div>
          <div className="value">{item.value}</div>
          <div
            className={`delta ${
              item.deltaClass ? item.deltaClass : ""
            }`}
          >
            {item.delta}
          </div>
        </div>
      ))}
    </div>
  );
}
