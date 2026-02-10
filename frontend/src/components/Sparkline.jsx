const buildPath = (values, width, height) => {
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = width / (values.length - 1);
  return values
    .map((value, index) => {
      const x = index * step;
      const y = height - ((value - min) / range) * (height - 6) - 3;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
};

export default function Sparkline({
  values,
  width = 240,
  height = 60,
}) {
  const path = buildPath(values, width, height);

  return (
    <svg
      className="sparkline"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
    >
      <path
        d={path}
        fill="none"
        strokeWidth="2"
        pathLength="100"
        className="sparkline-path"
      />
      <circle cx={width - 4} cy={height / 2} r="0" />
    </svg>
  );
}
