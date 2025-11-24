# --------- add/replace these imports near top of option_buy.py ----------
import requests
import logging
import traceback

# --------- Replace existing fetch_option_chain_playwright / fetch logic ----------
def _default_headers(cookie_header: str = None, ua: str = None):
    ua_fallback = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/127.0.0.0 Safari/537.36")
    h = {
        "User-Agent": ua or ua_fallback,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "Origin": "https://www.nseindia.com",
    }
    if cookie_header:
        h["Cookie"] = cookie_header
    return h

def fetch_chain_requests(symbol: str, max_retries: int = 3, timeout: int = 10) -> dict:
    """Try a simple requests-based fetch of the NSE option-chain API endpoint."""
    api_idx = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    api_equ = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    url = api_idx if symbol.upper() in {"NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"} else api_equ

    sess = requests.Session()
    sess.headers.update(_default_headers())

    for attempt in range(1, max_retries + 1):
        try:
            # First visit NSE home & option-chain pages to get cookies (gentle)
            try:
                sess.get("https://www.nseindia.com", timeout=timeout)
                sess.get("https://www.nseindia.com/option-chain", timeout=timeout)
            except Exception:
                # ignore; we still try the API endpoint
                pass

            r = sess.get(url, timeout=timeout)
            ctype = (r.headers.get("Content-Type") or "").lower()
            txt = r.text or ""
            logging.info(f"requests.fetch attempt={attempt} status={r.status_code} ctype={ctype} len={len(txt)}")
            if r.status_code == 200:
                # sometimes content-type isn't accurate; try json()
                try:
                    return r.json()
                except Exception:
                    try:
                        return json.loads(txt)
                    except Exception:
                        raise RuntimeError("requests: response not JSON-parsable")
        except Exception as e:
            logging.warning(f"requests.fetch attempt={attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(0.7 * attempt)
            else:
                break
    return None

def fetch_chain_playwright_if_available(symbol: str, engine: str = "chromium", headless: bool = True) -> dict:
    """Fallback: use Playwright only if it's importable and usable in the environment."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        logging.info("playwright not importable or available; skipping playwright fetch.")
        return None

    api_idx = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    api_equ = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    url = api_idx if symbol.upper() in {"NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"} else api_equ

    try:
        with sync_playwright() as p:
            browser_type = {"chromium": p.chromium, "firefox": p.firefox, "webkit": p.webkit}.get(engine, p.chromium)
            browser = browser_type.launch(headless=headless)
            context = browser.new_context(
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/127.0.0.0 Safari/537.36"),
                extra_http_headers={"Accept": "application/json, text/plain, */*", "Referer": "https://www.nseindia.com/option-chain"}
            )
            page = context.new_page()
            try:
                page.goto("https://www.nseindia.com", timeout=30000)
            except Exception:
                pass
            try:
                page.goto("https://www.nseindia.com/option-chain", timeout=30000)
                page.wait_for_timeout(800)
            except Exception:
                pass

            resp = context.request.get(url, timeout=60000)
            ctype = (resp.headers.get("content-type") or "").lower()
            txt = resp.text()
            if resp.ok and 'application/json' in ctype:
                data = resp.json()
            elif resp.ok and txt and not txt.lstrip().startswith('<'):
                try:
                    data = json.loads(txt)
                except Exception:
                    data = None
            else:
                data = None

            try:
                context.storage_state(path=".nse_storage_state.json")
            except Exception:
                pass
            context.close()
            browser.close()
            return data
    except Exception as e:
        logging.warning("playwright fetch failed: " + str(e) + "\n" + traceback.format_exc())
        return None

def fetch_chain(symbol: str) -> dict:
    """Unified fetcher: requests-first (fast), then Playwright fallback if available."""
    # First try requests
    data = fetch_chain_requests(symbol)
    if data:
        logging.info("fetch_chain: used requests path")
        return data

    # Next, try Playwright only if available
    data = fetch_chain_playwright_if_available(symbol)
    if data:
        logging.info("fetch_chain: used playwright path")
        return data

    raise RuntimeError("Unable to fetch NSE option chain via requests or Playwright. See logs for details.")
