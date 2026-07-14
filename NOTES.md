1.- Large classes that got sub-split during chunking can appear multiple times
  in top-K retrieval results (e.g., Session class showed up 3x for one
  query). Consider deduplicating by symbol_name+file_path before passing
  results to the agent, or before showing to the user.

  2."Currently rebuilding the full BM25 index in memory on every script run (_load_bm25_index) — fine for 358 chunks, but for a much larger repo this would need to be cached/persisted instead of rebuilt each time." 

  3.NOTES.md addition:
- Agent sometimes calls the same tool twice with an identical query in one 
  turn (seen with web_search). Doesn't break anything since MAX_TOOL_ITERATIONS 
  caps it, but wastes an API call. Possible fix later: dedupe identical tool 
  calls within a single turn before executing them.

  4."Model occasionally calls calculator for irrelevant line-counting math not requested by the user — doesn't break anything, but shows tool-calling isn't perfectly targeted."

  5.
- llama-3.1-8b-instant occasionally makes irrelevant tool calls (e.g. calling 
  calculator with a trivial/unrelated expression like "1+1" on a purely 
  conceptual question). This appears to be a smaller-model tool-judgment 
  limitation — llama-3.3-70b-versatile showed this far less often in earlier 
  testing. Tradeoff: 8b conserves token budget but has weaker tool-call 
  judgment. Worth revisiting model choice if token budget allows.

  6.- CONFIRMED in production test: indexing Flask's repo surfaced the RST 
  chunking limitation directly — .rst doc files without markdown-style 
  headings (e.g. appcontext.rst, shell.rst) fall back to blind 800-char 
  window splitting, producing large/imprecise chunks compared to the clean 
  AST-based code chunking. Code retrieval unaffected. Planned fix: detect 
  RST underline-style headings (===, ---, ~~~ under a title line) similar 
  to how markdown # headings are detected. Deferred post-deployment.

  7.NOTES.md addition:
- code_execution tool runs inside the app container, which doesn't have 
  third-party libraries like `requests` installed (only LumosAI's own deps). 
  When the agent tries to execute code that imports the analyzed library, 
  it will fail. Currently the agent seems to gracefully fall back to 
  context-based answering when this happens, but this is worth hardening — 
  e.g. catching the specific ImportError and telling the agent the execution 
  environment doesn't have that library available.