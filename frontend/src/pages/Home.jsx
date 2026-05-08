import { useCallback, useMemo, useState } from "react";
import axios from "axios";
import SearchBar from "../components/SearchBar";
import ResultCard from "../components/ResultCard";

const api = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 120000
});

function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState("");

  const doSearch = useCallback(async (searchText) => {
    if (!searchText) return;

    setLoading(true);
    setError("");
    setSearched(true);

    try {
      const response = await api.post("/search", { query: searchText });
      const incoming = response?.data?.sources || [];
      const modelAnswer = response?.data?.answer || "";
      setResults(incoming);
      setAnswer(modelAnswer);
      setQuery(searchText);
    } catch (err) {
      const isTimeout = err?.code === "ECONNABORTED";
      const message = isTimeout
        ? "Request timed out while generating the legal analysis. Please retry, or refine the query."
        : err?.response?.data?.detail ||
          "Unable to fetch results. Please check whether backend is running on localhost:8000.";
      setError(message);
      setResults([]);
      setAnswer("");
    } finally {
      setLoading(false);
    }
  }, []);

  const onDebouncedQueryChange = useCallback((value) => {
    setQuery(value.trim());
  }, []);

  const renderAnswer = useCallback((text) => {
    if (!text) return null;
    const lines = text.split("\n");
    return (
      <div className="space-y-2 text-[15px] leading-8 text-slate-200">
        {lines.map((line, index) => {
          const trimmed = line.trim();
          if (!trimmed) return <div key={`line-${index}`} className="h-1" />;

          const lower = trimmed.toLowerCase();
          const isHeading =
            lower.startsWith("summary:") ||
            lower.startsWith("legal reasoning:") ||
            lower.startsWith("key factors:") ||
            lower.startsWith("citations:") ||
            lower.startsWith("answer:") ||
            lower.startsWith("reasoning:");

          if (isHeading) {
            return (
              <h4 key={`line-${index}`} className="pt-2 text-base font-bold text-white">
                {trimmed}
              </h4>
            );
          }

          if (trimmed.startsWith("- ")) {
            const citationLine = /high court|supreme court|\(\s*.*\d{4}.*\)/i.test(trimmed);
            return (
              <p
                key={`line-${index}`}
                className={citationLine ? "pl-1 text-brand-100" : "pl-1"}
              >
                • {trimmed.slice(2)}
              </p>
            );
          }

          return <p key={`line-${index}`}>{trimmed}</p>;
        })}
      </div>
    );
  }, []);

  const stateView = useMemo(() => {
    if (!searched) {
      return (
        <div className="mt-14 rounded-2xl border border-slate-800 bg-slate-900/70 p-8 text-center text-slate-300 shadow-soft">
          Start with a legal query to get AI legal analysis with supporting cases.
        </div>
      );
    }

    if (loading) {
      return (
        <div className="mt-14 space-y-5">
          <div className="flex items-center justify-center gap-3 text-slate-300">
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-brand-500/40 border-t-brand-600" />
            Generating AI legal analysis...
          </div>
          <div className="h-32 animate-pulse rounded-2xl border border-slate-800 bg-slate-900/70 shadow-soft" />
        </div>
      );
    }

    if (error) {
      return (
        <div className="mt-14 rounded-2xl border border-red-900/70 bg-red-950/40 p-6 text-red-200">
          {error}
        </div>
      );
    }

    if (!answer && !results.length) {
      return (
        <div className="mt-14 rounded-2xl border border-slate-800 bg-slate-900/70 p-8 text-center text-slate-300 shadow-soft">
          No results found.
        </div>
      );
    }

    return (
      <div className="mt-8 space-y-7">
        <section className="rounded-2xl border border-slate-800 bg-slate-900/90 p-6 shadow-soft">
          <h3 className="mb-4 text-xl font-extrabold tracking-tight text-white">
            AI Legal Analysis
          </h3>
          {answer ? (
            renderAnswer(answer)
          ) : (
            <p className="text-slate-300">No synthesized answer generated.</p>
          )}
        </section>

        <div className="h-px w-full bg-slate-800" />

        <section>
          <h3 className="mb-4 text-lg font-bold text-white">Supporting Cases</h3>
          {!results.length ? (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 text-slate-300 shadow-soft">
              No supporting cases found for this query.
            </div>
          ) : (
            <div className="max-h-[52vh] space-y-4 overflow-y-auto pr-1">
              {results.map((result, index) => (
                <ResultCard
                  key={`${result?.metadata?.id || "case"}-${index}`}
                  result={result}
                  query={query}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    );
  }, [answer, error, loading, query, renderAnswer, results, searched]);

  return (
    <main className="min-h-screen px-4 py-10 sm:px-6">
      <div className="mx-auto flex min-h-[calc(100vh-5rem)] w-full max-w-6xl flex-col items-center">
        <header className="text-center">
          <h1 className="bg-gradient-to-r from-slate-100 via-white to-brand-100 bg-clip-text text-4xl font-extrabold tracking-tight text-transparent sm:text-6xl">
            Nyaya-Sahakari
          </h1>
          <p className="mt-3 text-base font-medium text-slate-300 sm:text-lg">
            Legal Intelligence Powered by AI
          </p>
        </header>

        <div className="mt-12 flex w-full justify-center">
          <div className="w-full max-w-4xl">
            <SearchBar
              query={query}
              loading={loading}
              onSearch={doSearch}
              onDebouncedQueryChange={onDebouncedQueryChange}
            />
          </div>
        </div>

        <section className="w-full max-w-5xl">{stateView}</section>
      </div>
    </main>
  );
}

export default Home;
