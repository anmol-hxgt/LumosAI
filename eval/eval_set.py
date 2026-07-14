"""
Hand-labeled evaluation set for LumosAI, built against the `requests` library
(test_repos/requests/src/requests).

Question categories (increasing difficulty for retrieval):
  - lookup: single-file, single-symbol questions — the easy case
  - multi_hop: answering requires connecting multiple files/symbols together
  - call_chain: tests whether retrieval surfaces the actual call sequence
                 across files (e.g. A calls B calls C)
  - architecture: tests whether retrieval surfaces a whole subsystem's worth
                   of related symbols, not just one function

Each entry has:
  - question: a natural-language question a real developer might ask
  - ground_truth: the reference answer (used for faithfulness/relevancy scoring)
  - expected_files: file(s) that SHOULD be retrieved for this question
  - category: one of the four above, used to break down results later

IMPORTANT: before running the real eval, spot-check a handful of these
against the actual source files to confirm the ground_truth is accurate.
Wrong ground truth = meaningless eval numbers.
"""

EVAL_SET = [
    # ------------------------------------------------------------------
    # Type 1 — Lookup (single file/symbol, the easy case)
    # ------------------------------------------------------------------
    {
        "question": "What class is used to persist parameters like cookies and auth across multiple requests?",
        "ground_truth": "The Session class, defined in sessions.py, persists cookies, authentication, headers, and other parameters across multiple requests.",
        "expected_files": ["sessions.py"],
        "category": "lookup",
    },
    {
        "question": "How does the library decide whether to strip the Authorization header when following a redirect?",
        "ground_truth": "The should_strip_auth method in the SessionRedirectMixin class (sessions.py) determines whether the Authorization header should be removed during a redirect, generally based on whether the redirect changes the hostname.",
        "expected_files": ["sessions.py"],
        "category": "lookup",
    },
    {
        "question": "What method prepares a Request object into a PreparedRequest before sending?",
        "ground_truth": "The prepare_request method on the Session class (sessions.py) merges session-level settings with the individual Request's settings and constructs a PreparedRequest object.",
        "expected_files": ["sessions.py"],
        "category": "lookup",
    },
    {
        "question": "What class manages cookies and provides a dict-like interface for them?",
        "ground_truth": "The RequestsCookieJar class in cookies.py provides a dictionary-like interface for accessing and modifying cookies.",
        "expected_files": ["cookies.py"],
        "category": "lookup",
    },
    {
        "question": "What is the base class that all transport adapters inherit from?",
        "ground_truth": "BaseAdapter, defined in adapters.py, is the base transport adapter class that other adapters like HTTPAdapter inherit from.",
        "expected_files": ["adapters.py"],
        "category": "lookup",
    },
    {
        "question": "Where are the custom exception classes for this library defined?",
        "ground_truth": "Custom exceptions such as RequestException and its subclasses are defined in exceptions.py.",
        "expected_files": ["exceptions.py"],
        "category": "lookup",
    },
    {
        "question": "What class represents the response returned after a request is made?",
        "ground_truth": "The Response class, defined in models.py, represents the server's response to an HTTP request.",
        "expected_files": ["models.py"],
        "category": "lookup",
    },
    {
        "question": "What data structure is used to implement case-insensitive HTTP headers?",
        "ground_truth": "CaseInsensitiveDict, defined in structures.py, is used to store HTTP headers in a way that treats header names as case-insensitive.",
        "expected_files": ["structures.py"],
        "category": "lookup",
    },
    # ------------------------------------------------------------------
    # Type 2 — Multi-hop reasoning (answer spans multiple files)
    # ------------------------------------------------------------------
    {
        "question": "How does a Session send a request from start to finish?",
        "ground_truth": "A Session.send() call takes a PreparedRequest (built via prepare_request in sessions.py using Request/PreparedRequest from models.py), selects the right transport adapter (adapters.py) via get_adapter, sends it through that adapter to get a Response (models.py), then dispatches any registered response hooks (hooks.py) before returning the Response.",
        "expected_files": ["sessions.py", "adapters.py", "models.py", "hooks.py"],
        "category": "multi_hop",
    },
    {
        "question": "How does connection pooling get used when a request is sent through a Session?",
        "ground_truth": "The Session holds transport adapters (adapters.py) mounted per URL prefix; HTTPAdapter wraps a urllib3 PoolManager that maintains connection pools, so when Session.send() dispatches a request via get_adapter, the underlying HTTPAdapter reuses pooled connections instead of opening a new one each time.",
        "expected_files": ["sessions.py", "adapters.py"],
        "category": "multi_hop",
    },
    {
        "question": "How does authentication get attached to a request before it is actually sent?",
        "ground_truth": "Authentication is attached via the auth parameter on Session or Request; PreparedRequest.prepare_auth (models.py) calls the auth handler (e.g. from auth.py) which modifies the request's headers before Session.send() transmits it via the chosen adapter.",
        "expected_files": ["models.py", "sessions.py", "auth.py"],
        "category": "multi_hop",
    },
    # ------------------------------------------------------------------
    # Type 3 — Call chain (does retrieval surface the actual call sequence)
    # ------------------------------------------------------------------
    {
        "question": "How are response hooks executed after an HTTP request completes?",
        "ground_truth": "Session.send() (sessions.py) calls dispatch_hook (hooks.py) with the 'response' event and the Response object, which runs any user-registered callback functions and lets them optionally replace the response.",
        "expected_files": ["sessions.py", "hooks.py"],
        "category": "call_chain",
    },
    {
        "question": "What happens internally when a response indicates a redirect needs to be followed?",
        "ground_truth": "Session.resolve_redirects (sessions.py, part of SessionRedirectMixin) checks the response's Location header, builds the next request via rebuild_auth and related helpers, and yields subsequent Response objects (models.py) until no further redirect is indicated.",
        "expected_files": ["sessions.py", "models.py"],
        "category": "call_chain",
    },
    # ------------------------------------------------------------------
    # Type 4 — Architecture (whole subsystem, many related symbols)
    # ------------------------------------------------------------------
    {
        "question": "Explain the redirect handling flow inside the Requests library.",
        "ground_truth": "Redirect handling lives in SessionRedirectMixin (sessions.py). resolve_redirects drives the loop of following redirects; should_strip_auth decides whether to drop the Authorization header when the redirect target's host differs; rebuild_auth and related rebuild_* methods reconstruct headers/cookies/proxies for the new request at each hop.",
        "expected_files": ["sessions.py"],
        "category": "architecture",
    },
    {
        "question": "Explain how errors during a request get turned into raised exceptions.",
        "ground_truth": "exceptions.py defines a hierarchy rooted at RequestException (subclassing IOError) with specific subclasses like ConnectionError, Timeout, and HTTPError. Adapters (adapters.py) catch lower-level urllib3 errors and re-raise them as these Requests-specific exceptions, and Response.raise_for_status (models.py) raises HTTPError for 4xx/5xx status codes.",
        "expected_files": ["exceptions.py", "adapters.py", "models.py"],
        "category": "architecture",
    },
]


if __name__ == "__main__":
    print(f"Eval set has {len(EVAL_SET)} questions.\n")
    from collections import Counter

    counts = Counter(item["category"] for item in EVAL_SET)
    for category, count in counts.items():
        print(f"  {category}: {count}")
    print()
    for i, item in enumerate(EVAL_SET, 1):
        print(f"{i}. [{item['category']}] {item['question']}")
        print(f"   expected: {item['expected_files']}")