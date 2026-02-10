export default function StatusPill({ status = "idle", label }) {
  return (
    <span className={`status-pill ${status}`}>
      <span className="pulse" />
      {label}
    </span>
  );
}
