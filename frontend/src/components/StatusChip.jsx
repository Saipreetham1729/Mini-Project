const LABELS = {
  supported: "Supported",
  partially_supported: "Partially supported",
  unsupported: "Unsupported",
};

export default function StatusChip({ status }) {
  if (!status) return null;
  return <span className={`chip ${status}`}>{LABELS[status] || status}</span>;
}
