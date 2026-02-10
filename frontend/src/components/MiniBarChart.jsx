export default function MiniBarChart({ bars, width = 420, height = 140 }) {
  const maxAbs = Math.max(
    1,
    ...bars.map((bar) => Math.abs(bar.value))
  );
  const baseY = height / 2;
  const barWidth = width / (bars.length * 1.35);
  const gap = barWidth * 0.35;

  return (
    <svg
      className="bar-chart"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
    >
      <line
        x1="0"
        y1={baseY}
        x2={width}
        y2={baseY}
        className="bar-baseline"
      />
      {bars.map((bar, index) => {
        const magnitude = (Math.abs(bar.value) / maxAbs) * (height / 2 - 8);
        const x = index * (barWidth + gap) + gap;
        const y = bar.value >= 0 ? baseY - magnitude : baseY;
        const color = bar.value >= 0 ? "var(--accent)" : "var(--muted)";

        return (
          <g key={bar.label}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={Math.max(4, magnitude)}
              rx="2"
              className="bar-rect"
              style={{ fill: color, opacity: 0.85 }}
            />
            <text x={x + barWidth / 2} y={height - 4} textAnchor="middle">
              {bar.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
