export default function Panel({
  id,
  title,
  active,
  onSelect,
  children,
  className = "",
}) {
  return (
    <div
      className={`panel ${active ? "active" : ""} ${className}`}
      onClick={() => onSelect(id)}
    >
      <h3>{title}</h3>
      {children}
    </div>
  );
}
