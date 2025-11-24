# api_server.py (lazy-import, robust, railway-friendly)
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import importlib
import logging
import os
import traceback
from typing import Any, Dict, Optional

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("api_server")

app = FastAPI(title="Option Buy API (lazy import)", version="1.0.0")

# Keep state about the imported option module and any import errors
OPTION_MODULE_NAME = os.environ.get("OPTION_MODULE", "option_buy_api")  # change via env if needed
_option_module = None
_option_import_error: Optional[str] = None

def lazy_import_option_module():
    global _option_module, _option_import_error
    if _option_module is not None or _option_import_error is not None:
        return
    try:
        logger.info("Attempting to import option module '%s'...", OPTION_MODULE_NAME)
        _option_module = importlib.import_module(OPTION_MODULE_NAME)
        logger.info("Imported module '%s' from %s", OPTION_MODULE_NAME, getattr(_option_module, "__file__", "<unknown>"))
    except Exception as e:
        _option_import_error = traceback.format_exc()
        logger.error("Failed to import option module '%s': %s", OPTION_MODULE_NAME, e)
        logger.debug("%s", _option_import_error)

@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Lightweight health endpoint that returns quickly.
    Will not try to import heavy dependencies.
    """
    # Attempt import in background but don't block health response (we still call but it's quick)
    try:
        lazy_import_option_module()
    except Exception:
        # ignore - lazy_import_option_module stores error
        pass

    return {
        "status": "ok",
        "option_module": OPTION_MODULE_NAME,
        "module_loaded": bool(_option_module),
        "module_file": getattr(_option_module, "__file__", None) if _option_module else None,
        "import_error_present": bool(_option_import_error),
    }

@app.get("/debug")
def debug() -> JSONResponse:
    """
    Returns debug info: available attributes on the loaded module and import traceback if any.
    """
    lazy_import_option_module()
    attrs = []
    if _option_module:
        attrs = sorted([n for n in dir(_option_module) if not n.startswith("_")])
    return JSONResponse(
        status_code=200,
        content={
            "option_module": OPTION_MODULE_NAME,
            "module_loaded": bool(_option_module),
            "module_file": getattr(_option_module, "__file__", None) if _option_module else None,
            "available_attrs": attrs,
            "import_error": _option_import_error,
            "env": {
                "GS_CREDS_JSON": os.environ.get("GS_CREDS_JSON"),
                "GOOGLE_SHEET_NAME": os.environ.get("GOOGLE_SHEET_NAME"),
                "CAPITAL": os.environ.get("CAPITAL"),
                "RISK_PER_TRADE_PCT": os.environ.get("RISK_PER_TRADE_PCT"),
            },
        },
    )

@app.get("/analyze")
def analyze(symbol: str = Query(..., description="Underlying symbol, e.g. HDFCBANK or NIFTY"),
            log_to_sheet: bool = Query(False, description="If true, attempt Google Sheets logging")) -> JSONResponse:
    """
    Analyze endpoint: lazily imports the option module and calls its functions.
    This handler uses defensive calling to accommodate slight variations in function names.
    """

    lazy_import_option_module()

    if _option_import_error:
        # Import failed earlier — return the traceback so you can see the reason in browser
        return JSONResponse(status_code=500, content={
            "error": "Failed to import option module",
            "import_error": _option_import_error,
        })

    if _option_module is None:
        return JSONResponse(status_code=500, content={"error": "Option module not loaded. Check logs."})

    # find a fetch function
    fetch_fn = None
    for candidate in ("fetch_option_chain_playwright", "fetch_chain_playwright_if_available", "fetch_chain", "fetch_chain_requests"):
        if hasattr(_option_module, candidate):
            fetch_fn = getattr(_option_module, candidate)
            logger.info("Using fetch function: %s", candidate)
            break

    if fetch_fn is None or not hasattr(_option_module, "analyze_option_chain") or not hasattr(_option_module, "select_candidate_strikes"):
        # Missing analysis-level functionality — return available attrs to help debugging
        available = sorted([n for n in dir(_option_module) if not n.startswith("_")])
        logger.error("Module missing required analyze/select functions. Available attrs: %s", available)
        return JSONResponse(status_code=500, content={
            "error": "Option module does not expose required functions (fetch/analyze/select).",
            "available_attrs": available
        })

    analyze_fn = getattr(_option_module, "analyze_option_chain")
    select_fn = getattr(_option_module, "select_candidate_strikes")
    log_fn = getattr(_option_module, "log_to_google_sheet", None)

    # Prepare config values (try module-level first)
    CAPITAL = getattr(_option_module, "CAPITAL", float(os.environ.get("CAPITAL", 10000)))
    RISK_PER_TRADE_PCT = getattr(_option_module, "RISK_PER_TRADE_PCT", float(os.environ.get("RISK_PER_TRADE_PCT", 1.0)))
    GS_CREDS_JSON = getattr(_option_module, "GS_CREDS_JSON", os.environ.get("GS_CREDS_JSON"))
    GOOGLE_SHEET_NAME = getattr(_option_module, "GOOGLE_SHEET_NAME", os.environ.get("GOOGLE_SHEET_NAME", "OptionBuyLog"))

    # temporarily override SYMBOL if module exposes it
    original_symbol = None
    if hasattr(_option_module, "SYMBOL"):
        original_symbol = getattr(_option_module, "SYMBOL")
        try:
            setattr(_option_module, "SYMBOL", symbol.upper())
        except Exception:
            logger.debug("Could not set SYMBOL on module; continuing.")

    try:
        logger.info("Fetching option chain for %s", symbol.upper())
        # call fetch function defensively
        data = None
        try:
            # preferred signature: fetch_fn(symbol)
            data = fetch_fn(symbol.upper())
        except TypeError:
            # try alternate common signatures
            try:
                data = fetch_fn(symbol.upper(), "chromium", True, None)
            except Exception:
                try:
                    data = fetch_fn(symbol.upper(), None)
                except Exception as e:
                    logger.exception("Fetch function failed with multiple signatures: %s", e)
                    raise

        if not data:
            return JSONResponse(status_code=200, content={"symbol": symbol.upper(), "candidates": [], "message": "Fetch returned no data."})

        df = analyze_fn(data, symbol.upper())
        if df is None or getattr(df, "empty", False):
            return JSONResponse(status_code=200, content={"symbol": symbol.upper(), "candidates": [], "message": "No valid option data parsed."})

        candidates = select_fn(df, CAPITAL, RISK_PER_TRADE_PCT)
        if not candidates:
            return JSONResponse(status_code=200, content={"symbol": symbol.upper(), "candidates": [], "message": "No candidates found after filtering."})

        # optionally log to Google Sheets (best-effort)
        sheet_warning = None
        if log_to_sheet:
            try:
                if log_fn and (GS_CREDS_JSON and os.path.exists(GS_CREDS_JSON)):
                    log_fn(GS_CREDS_JSON, GOOGLE_SHEET_NAME, candidates)
                else:
                    sheet_warning = "Google Sheets logging requested but creds or log function missing."
            except Exception as e:
                logger.exception("Google Sheets logging failed: %s", e)
                sheet_warning = f"Google Sheets logging failed: {e}"

        response = {"symbol": symbol.upper(), "candidates": candidates}
        if sheet_warning:
            response["warning"] = sheet_warning

        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Unhandled exception in /analyze: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})
    finally:
        # restore original symbol
        try:
            if original_symbol is not None and hasattr(_option_module, "SYMBOL"):
                setattr(_option_module, "SYMBOL", original_symbol)
        except Exception:
            logger.exception("Failed to restore SYMBOL on option module.")
