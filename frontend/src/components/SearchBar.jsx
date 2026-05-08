import { useEffect, useState } from "react";

function SearchBar({ query, loading, onSearch, onDebouncedQueryChange }) {
  const [value, setValue] = useState(query);

  useEffect(() => {
    const timer = setTimeout(() => {
      onDebouncedQueryChange(value);
    }, 300);

    return () => clearTimeout(timer);
  }, [value, onDebouncedQueryChange]);

  const handleSubmit = (event) => {
    event.preventDefault();
    onSearch(value.trim());
  };

  return (
    <form onSubmit={handleSubmit} className="mx-auto w-full max-w-3xl">
      <div className="rounded-3xl border border-slate-700/70 bg-slate-900/80 p-2 shadow-soft backdrop-blur-xl transition-all duration-300 hover:border-brand-500/60">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <input
            type="text"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            className="h-14 flex-1 rounded-2xl border-0 bg-transparent px-5 text-lg text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-0"
            placeholder="Search legal cases, bail decisions, IPC sections..."
          />
          <button
            type="submit"
            disabled={loading || !value.trim()}
            className="inline-flex h-12 min-w-32 items-center justify-center gap-2 rounded-2xl border border-slate-300/80 bg-gradient-to-b from-white/95 to-slate-100 px-6 font-semibold text-slate-900 shadow-sm transition-all duration-300 hover:from-white hover:to-slate-50 hover:shadow-md disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? (
              <>
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/50 border-t-white" />
                Searching
              </>
            ) : (
              "Search"
            )}
          </button>
        </div>
      </div>
    </form>
  );
}

export default SearchBar;
