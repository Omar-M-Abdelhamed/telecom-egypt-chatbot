"""
scraper.py
----------
Scrapes the Telecom Egypt website (te.eg) and returns a list of
{"url": ..., "text": ...} dicts ready to be stored in ChromaDB.

The scraper only runs once — subsequent runs detect the persisted
ChromaDB and skip scraping entirely (handled in vectorstore.py).
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL = "https://te.eg"

# Seed URLs — covers all major sections including residential, business, offers
SEED_URLS = [
    # Homepage
    "https://te.eg",
    "https://te.eg/wps/portal/te/",
    # Residential / Personal
    "https://te.eg/wps/portal/te/residential",
    "https://te.eg/wps/portal/te/residential/internet",
    "https://te.eg/wps/portal/te/residential/internet/home-internet",
    "https://te.eg/wps/portal/te/residential/packages",
    "https://te.eg/wps/portal/te/residential/landline",
    "https://te.eg/wps/portal/te/residential/fiber",
    "https://te.eg/wps/portal/te/residential/tv",
    # Business
    "https://te.eg/wps/portal/te/business",
    "https://te.eg/wps/portal/te/business/internet",
    "https://te.eg/wps/portal/te/business/packages",
    # Offers
    "https://te.eg/wps/portal/te/offers",
    # Legacy www. variants (the site sometimes uses both)
    "https://www.te.eg/wps/portal/te/Personal",
    "https://www.te.eg/wps/portal/te/Personal/Internet",
    "https://www.te.eg/wps/portal/te/Personal/ADSL",
    "https://www.te.eg/wps/portal/te/Personal/Fiber",
    "https://www.te.eg/wps/portal/te/Business",
]

MAX_PAGES     = 150         # increased — we want deeper coverage
MAX_DEPTH     = 3           # crawl up to 3 levels deep from each seed
REQUEST_DELAY = 0.8         # seconds between requests
REQUEST_TIMEOUT = 20        # seconds per request

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Priority URL patterns — these are fetched before generic links
PRIORITY_PATTERNS = [
    "/residential", "/personal", "/business", "/offers",
    "/internet", "/fiber", "/adsl", "/landline", "/packages",
    "/price", "/plan", "/bundle", "/service",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_valid_te_url(url: str) -> bool:
    """Return True only for te.eg HTML pages we actually want to scrape."""
    parsed = urlparse(url)
    # Must be on te.eg domain (with or without www.)
    netloc = parsed.netloc.lower()
    if netloc and not ("te.eg" in netloc):
        return False
    # Skip binary / media / tracking files
    skip_ext = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
        ".mp4", ".zip", ".exe", ".doc", ".docx", ".xls",
        ".xlsx", ".ppt", ".pptx", ".css", ".js", ".ico",
        ".woff", ".woff2", ".ttf", ".eot",
    )
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in skip_ext):
        return False
    # Skip login / auth / logout pages
    skip_patterns = ["login", "logout", "signin", "signup", "register", "auth"]
    if any(p in path_lower for p in skip_patterns):
        return False
    return True


def _extract_text(soup: BeautifulSoup, url: str) -> str:
    """
    Pull ALL visible text from a parsed HTML page.
    Deliberately keeps pricing tables, headings, and list items.
    """
    # Remove truly noisy tags (keep nav/aside since they may have menu prices)
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Extract page title as a strong signal
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Extract all headings with their level for context
    headings = []
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = h.get_text(" ", strip=True)
        if text:
            headings.append(text)

    # Extract all table data (prices live in tables)
    tables_text = []
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            tables_text.append("\n".join(rows))

    # Full page text
    body_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    # Filter out very short meaningless lines (single chars, numbers only)
    lines = [l for l in lines if len(l) > 2]

    # Combine: title + headings + tables + body
    parts = []
    if title:
        parts.append(f"PAGE TITLE: {title}")
    if headings:
        parts.append("HEADINGS:\n" + "\n".join(headings))
    if tables_text:
        parts.append("TABLES:\n" + "\n\n".join(tables_text))
    parts.append("CONTENT:\n" + "\n".join(lines))

    return "\n\n".join(parts)


def _extract_links(soup: BeautifulSoup, current_url: str) -> tuple[list[str], list[str]]:
    """
    Return (priority_links, regular_links) found on the page.
    Priority links match key service/pricing patterns and are crawled first.
    """
    priority = []
    regular  = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue

        absolute = urljoin(current_url, href).split("#")[0].rstrip("/")
        if not absolute or not _is_valid_te_url(absolute):
            continue

        path_lower = urlparse(absolute).path.lower()
        if any(p in path_lower for p in PRIORITY_PATTERNS):
            priority.append(absolute)
        else:
            regular.append(absolute)

    return priority, regular


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_te_website(
    seed_urls: list[str] = SEED_URLS,
    max_pages: int = MAX_PAGES,
    max_depth: int = MAX_DEPTH,
    verbose: bool = True,
) -> list[dict]:
    """
    BFS-crawl te.eg starting from seed_urls, up to max_depth levels deep.

    Returns
    -------
    list of {"url": str, "text": str}
    """
    # Queue items: (url, depth)
    visited: set[str]   = set()
    # Seed URLs start at depth 0; prioritise seeds first
    queue: list[tuple[str, int]] = [(url.rstrip("/"), 0) for url in seed_urls]
    results: list[dict] = []

    failed  = 0
    skipped = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"[Scraper] Starting crawl — max {max_pages} pages, depth ≤ {max_depth}")
    print(f"[Scraper] Seeds: {len(seed_urls)} URLs")

    while queue and len(visited) < max_pages:
        url, depth = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        try:
            if verbose:
                print(f"[Scraper] [{len(visited)}/{max_pages}] depth={depth} {url}")

            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            # Only process HTML pages
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                skipped += 1
                continue

            soup = BeautifulSoup(response.text, "lxml")
            text = _extract_text(soup, url)

            # Keep pages with meaningful content (at least 150 chars)
            if len(text) >= 150:
                results.append({"url": url, "text": text})
                if verbose:
                    print(f"           → saved {len(text):,} chars")
            else:
                skipped += 1
                if verbose:
                    print(f"           → skipped (too short: {len(text)} chars)")

            # Discover more links if we haven't hit max depth
            if depth < max_depth:
                priority_links, regular_links = _extract_links(soup, url)

                # Deduplicate and filter already-visited
                new_priority = [l for l in dict.fromkeys(priority_links) if l not in visited]
                new_regular  = [l for l in dict.fromkeys(regular_links)  if l not in visited]

                # Priority links go to the FRONT of the queue
                front_items = [(l, depth + 1) for l in new_priority]
                back_items  = [(l, depth + 1) for l in new_regular]
                queue = front_items + queue + back_items

            time.sleep(REQUEST_DELAY)

        except requests.exceptions.HTTPError as e:
            failed += 1
            if verbose:
                print(f"[Scraper] HTTP error {e.response.status_code} — {url}")
        except requests.exceptions.RequestException as e:
            failed += 1
            if verbose:
                print(f"[Scraper] Connection error — {url}: {e}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[Scraper] DONE")
    print(f"  Pages scraped (with content): {len(results)}")
    print(f"  Pages visited (total):        {len(visited)}")
    print(f"  Pages skipped (no content):   {skipped}")
    print(f"  Pages failed  (errors):       {failed}")
    print(f"  Total text collected:         {sum(len(r['text']) for r in results):,} chars")
    print("=" * 60 + "\n")

    return results
