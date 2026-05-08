import { useMemo, useState } from "react";

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightText(text, query) {
  if (!query) return text;

  const tokens = query
    .toLowerCase()
    .split(/\s+/)
    .filter((item) => item.length > 2);

  if (!tokens.length) return text;

  const pattern = new RegExp(`(${tokens.map(escapeRegExp).join("|")})`, "gi");
  const parts = text.split(pattern);

  return parts.map((part, index) => {
    const match = tokens.includes(part.toLowerCase());
    return match ? (
      <mark key={`${part}-${index}`} className="rounded bg-brand-100 px-1 text-slate-900">
        {part}
      </mark>
    ) : (
      <span key={`${part}-${index}`}>{part}</span>
    );
  });
}

function ResultCard({ result, query }) {
  const [expanded, setExpanded] = useState(false);

  const preview = useMemo(() => {
    if (!result?.text) return "";
    if (expanded) return result.text;
    return result.text.length > 300 ? `${result.text.slice(0, 300)}...` : result.text;
  }, [expanded, result]);

  const court = result?.metadata?.court || "Unknown Court";
  const year = result?.metadata?.year || "N/A";
  const outcome = result?.metadata?.outcome || "N/A";

  return (
    <article className="rounded-2xl border border-slate-700/70 bg-slate-900/80 p-5 shadow-soft backdrop-blur-sm transition hover:border-brand-500/40">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="rounded-full border border-slate-600/80 bg-slate-800/70 px-3 py-1 text-xs font-medium text-slate-200">
          {court}
        </span>
        <span className="rounded-full border border-slate-600/80 bg-slate-800/70 px-3 py-1 text-xs font-medium text-slate-200">
          Year: {year}
        </span>
        <span className="rounded-full border border-slate-600/80 bg-slate-800/70 px-3 py-1 text-xs font-medium text-slate-200">
          Outcome: {outcome}
        </span>
      </div>

      <p className="whitespace-pre-wrap text-sm leading-7 text-slate-200">
        {highlightText(preview, query)}
      </p>

      {result?.text?.length > 300 && (
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="mt-3 text-sm font-semibold text-brand-600 transition hover:text-brand-700"
        >
          {expanded ? "Collapse" : "Expand"}
        </button>
      )}
    </article>
  );
}

export default ResultCard;
