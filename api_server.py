# api_server.py
"""
Robust API wrapper for the option-buy script.

Behavior:
 - Prefer importing CLI module 'option_buy' (contains analyze/select/log functions).
 - Fallback order: option_buy -> option_buy_api -> option_buy_cli
 - Exposes /health, /debug, /analyze endpoints with good diagnostics for Railway logs.
"""
import importlib
import logging
import os
import traceback
from typing import Dict, Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# logging (Railway captures stdout/stderr)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("api_server")

# Try import order: CLI (full logic) first, then API-only module
TRY_ORDER = ("option_buy", "option_buy_api", "option_buy_cli", "option_buy_api")
loaded_module = None
import_errors = {}

for name in TRY_ORDER:
    try:
        loaded_module = importlib.import_module(name)
        logger.info("Imported module '%s' (file=%s)", name, getattr(loaded_module, "__file__", "<unknown>"))
        break
    except Exception as e:
        import_errors[name] = traceback.format_exc()
        logger.debug("Import failed for %s: %s", name, import_errors[name])

if loaded_module is None:
    logger.error("Failed to import any option module. Tried: %s", TRY_ORDER)

# helper to check available attrs on loaded module
def _has(attr: str) -> bool:
    return (loaded_module is not None) and hasattr(loaded_module, attr)

app = FastAPI(title="Option Buy API (adaptive)", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    info = {"status": "ok"}
    if loaded_module:
        info["module"] = loaded_module.__name__
        info["module_file"] = getattr(loaded_module, "__file__", None)
    else:
        info["module"] = None
        info["note"] = "No option module imported; check logs."
    return info


@app.get("/debug")
def debug() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "imported_module": loaded_module.__name__ if loaded_module else None,
            "module_file": getattr(loaded_module, "__file__", None) if loaded_module else None,
            "available_attrs": sorted([n for n in dir(loaded_module) if not n.startswith("_")]) if loaded_module else [],
            "import_errors": {k: ("present" if v else None) for k, v in import_errors.items()},
            "env_sample": {
                "GS_CREDS_JSON": os.environ.get("GS_CREDS_JSON"),
                "GOOGLE_SHEET_NAME": os.environ.get("GOOGLE_SHEET_NAME"),
            },
        },
    )


@app.get("/analyze")
def analyze(
    symbol: str = Query(..., description="Underlying symbol, e.g. HDFCBANK or NIFTY"),
    log_to_sheet: bool = Query(False, description="If true, attempt to log results to Google Sheet."),
) -> JSONResponse:
    # Ensure we have something imported
    if loaded_module is None:
        msg = f"No option module imported. Tried: {TRY_ORDER}. See logs."
        logger.error(msg)
        return JSONResponse(status_code=500, content={"error": msg, "import_errors": import_errors})

    # Prefer the CLI-style functions if present
    # Required for full behavior: fetch_option_chain_playwright (or fetch_chain), analyze_option_chain, select_candidate_strikes
    # We'll attempt to use the best available functions and give clear error messages if missing.
    fetch_fn = None
    analyze_fn = None
    select_fn = None
    log_fn = getattr(loaded_module, "log_to_google_sheet", None)

    # 1) fetch function
    if _has("fetch_option_chain_playwright"):
        fetch_fn = getattr(loaded_module, "fetch_option_chain_playwright")
        logger.info("Using fetch_option_chain_playwright from %s", loaded_module.__name__)
    elif _has("fetch_chain_playwright_if_available"):
        fetch_fn = getattr(loaded_module, "fetch_chain_playwright_if_available")
        logger.info("Using fetch_chain_playwright_if_available from %s", loaded_module.__name__)
    elif _has("fetch_chain"):
        # fetch_chain in some modules accepts (symbol, engine, headless, log_fh) — we will call with sensible defaults.
        fetch_fn = getattr(loaded_module, "fetch_chain")
        logger.info("Using fetch_chain from %s (will call with defaults)", loaded_module.__name__)
    elif _has("fetch_chain_requests"):
        fetch_fn = getattr(loaded_module, "fetch_chain_requests")
        logger.info("Using fetch_chain_requests from %s", loaded_module.__name__)
    else:
        logger.warning("No fetch function found in module %s", loaded_module.__name__)

    # 2) analysis & selection functions - prefer CLI module that contains these
    if _has("analyze_option_chain"):
        analyze_fn = getattr(loaded_module, "analyze_option_chain")
        logger.info("Using analyze_option_chain from %s", loaded_module.__name__)
    if _has("select_candidate_strikes"):
        select_fn = getattr(loaded_module, "select_candidate_strikes")
        logger.info("Using select_candidate_strikes from %s", loaded_module.__name__)

    # If loaded_module didn't provide analyze/select, try to import 'option_buy' explicitly as a source of analysis
    if (analyze_fn is None or select_fn is None) and loaded_module.__name__ != "option_buy":
        try:
            alt = importlib.import_module("option_buy")
            if analyze_fn is None and hasattr(alt, "analyze_option_chain"):
                analyze_fn = getattr(alt, "analyze_option_chain")
                logger.info("Falling back to analyze_option_chain from option_buy")
            if select_fn is None and hasattr(alt, "select_candidate_strikes"):
                select_fn = getattr(alt, "select_candidate_strikes")
                logger.info("Falling back to select_candidate_strikes from option_buy")
            # also use its constants if available
            if not hasattr(loaded_module, "SYMBOL") and hasattr(alt, "SYMBOL"):
                logger.info("option_buy provides SYMBOL; will use that when needed.")
        except Exception:
            logger.debug("Could not import fallback module 'option_buy' for analysis/select: %s", traceback.format_exc())

    # Fail early if we don't have the core pieces
    missing = []
    if fetch_fn is None:
        missing.append("fetch function (fetch_option_chain_playwright / fetch_chain / fetch_chain_playwright_if_available / fetch_chain_requests)")
    if analyze_fn is None:
        missing.append("analyze_option_chain")
    if select_fn is None:
        missing.append("select_candidate_strikes")
    if missing:
        msg = f"Module does not expose required functions: {missing}. Available attrs: {sorted([n for n in dir(loaded_module) if not n.startswith('_')])}"
        logger.error(msg)
        return JSONResponse(status_code=500, content={"error": msg})

    # Prepare config values (prefer module-level values, otherwise environment)
    CAPITAL = getattr(loaded_module, "CAPITAL", None) or float(os.environ.get("CAPITAL", 10000))
    RISK_PCT = getattr(loaded_module, "RISK_PER_TRADE_PCT", None) or float(os.environ.get("RISK_PER_TRADE_PCT", 1.0))
    GS_CREDS = getattr(loaded_module, "GS_CREDS_JSON", None) or os.environ.get("GS_CREDS_JSON")
    SHEET_NAME = getattr(loaded_module, "GOOGLE_SHEET_NAME", None) or os.environ.get("GOOGLE_SHEET_NAME", "OptionBuyLog")

    # Temporarily set module SYMBOL if present to keep CLI behaviour consistent
    original_symbol = None
    if hasattr(loaded_module, "SYMBOL"):
        original_symbol = getattr(loaded_module, "SYMBOL")
        try:
            setattr(loaded_module, "SYMBOL", symbol.upper())
        except Exception:
            logger.debug("Could not overwrite SYMBOL on loaded module; continuing without overwrite.")

    try:
        logger.info("Fetching option chain for %s", symbol.upper())
        # call fetch_fn; adapt if it's fetch_chain which expects extra args
        data = None
        try:
            # best-effort calling conventions:
            # - ideal: fetch_fn(symbol)
            # - fallback: fetch_fn(symbol, engine='chromium', headless=True, log_fh=None)
            # - fallback: fetch_chain_requests(symbol, log_fh, ua=None)
            try:
                data = fetch_fn(symbol.upper())
            except TypeError:
                # try common alternate signatures
                try:
                    data = fetch_fn(symbol.upper(), "chromium", True, None)
                except TypeError:
                    try:
                        data = fetch_fn(symbol.upper(), None)
                    except Exception:
                        # last resort: call with only symbol and hope
                        data = fetch_fn(symbol.upper())
        except Exception as e:
            logger.exception("Fetch function error: %s", e)
            raise

        if not data:
            msg = "Fetch returned no data."
            logger.warning(msg)
            return JSONResponse(status_code=200, content={"symbol": symbol.upper(), "candidates": [], "message": msg})

        df = analyze_fn(data, symbol.upper())
        if df is None or getattr(df, "empty", False):
            msg = "No valid option data parsed."
            logger.info(msg)
            return JSONResponse(status_code=200, content={"symbol": symbol.upper(), "candidates": [], "message": msg})

        candidates = select_fn(df, CAPITAL, RISK_PCT)
        if not candidates:
            msg = "No candidates found after filtering. Try relaxing MIN_OI or MIN_VOLUME."
            logger.info(msg)
            return JSONResponse(status_code=200, content={"symbol": symbol.upper(), "candidates": [], "message": msg})

        # Optional logging to Google Sheets (best-effort)
        sheet_warning = None
        if log_to_sheet:
            try:
                if log_fn and (GS_CREDS or os.path.exists(GS_CREDS if GS_CREDS else "")):
                    log_fn(GS_CREDS, SHEET_NAME, candidates)
                    logger.info("Logged to Google Sheet (sheet=%s)", SHEET_NAME)
                else:
                    sheet_warning = "Google Sheets logging requested but log function or creds missing."
                    logger.warning(sheet_warning)
            except Exception as e:
                logger.exception("Google Sheets logging failed: %s", e)
                sheet_warning = f"Failed to log to Google Sheet: {e}"

        resp = {"symbol": symbol.upper(), "candidates": candidates}
        if sheet_warning:
            resp["warning"] = sheet_warning

        logger.info("Returning %d candidates for %s", len(candidates), symbol.upper())
        return JSONResponse(status_code=200, content=resp)

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Unhandled error in /analyze: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})
    finally:
        # restore original symbol
        try:
            if original_symbol is not None and hasattr(loaded_module, "SYMBOL"):
                setattr(loaded_module, "SYMBOL", original_symbol)
        except Exception:
            logger.exception("Failed to restore SYMBOL on loaded module.")
