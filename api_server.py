# api_server.py
"""
Robust API wrapper for the option-buy script.

- Tries to import option_buy_api (preferred for API deployments), then option_buy, then option_buy_cli.
- Avoids import-time crashes by keeping module import separate from attribute lookup.
- Exposes /health, /analyze and /debug endpoints.
- Provides helpful diagnostics in error responses and logs for Railway.
"""
import importlib
import logging
import os
import traceback
from typing import Any, Dict, List

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# Configure logging - Railway will show stdout/stderr logs
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("api_server")

# Attempt to import preferred option module(s)
PREFERRED_MODULES = ("option_buy_api", "option_buy", "option_buy_cli", "option_buy")
option_buy_module = None
import_errors = {}

for modname in PREFERRED_MODULES:
    try:
        option_buy_module = importlib.import_module(modname)
        logger.info("Imported module '%s' from %s", modname, getattr(option_buy_module, "__file__", "<unknown>"))
        break
    except Exception as e:
        import_errors[modname] = repr(e)
        logger.debug("Failed to import %s: %s", modname, traceback.format_exc())

if option_buy_module is None:
    logger.error("Unable to import any of the option modules: %s. Import errors: %s", PREFERRED_MODULES, import_errors)

# Helper to fetch attribute or raise a descriptive RuntimeError
def _get_attr(name: str):
    if option_buy_module is None:
        raise RuntimeError(f"No option module imported. Tried: {PREFERRED_MODULES}. See logs.")
    if not hasattr(option_buy_module, name):
        avail = sorted([n for n in dir(option_buy_module) if not n.startswith("_")])
        raise RuntimeError(f"Attribute '{name}' not found in module {option_buy_module.__name__}. Available: {avail}")
    return getattr(option_buy_module, name)

# Safely attempt to bind commonly used callables/vars. If missing, keep server up and return helpful messages in /analyze.
_bound = {}
for name in (
    "fetch_option_chain_playwright",
    "analyze_option_chain",
    "select_candidate_strikes",
    "log_to_google_sheet",
    "SYMBOL",
    "CAPITAL",
    "RISK_PER_TRADE_PCT",
    "GOOGLE_SHEET_NAME",
    "GS_CREDS_JSON",
):
    try:
        _bound[name] = _get_attr(name)
    except Exception as e:
        _bound[name] = None
        logger.warning("Could not bind '%s': %s", name, str(e))

app = FastAPI(title="Option Buy API (robust)", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Simple health check. Returns ok even if the option module is missing,
    but health response will include module import status in headers/body.
    """
    status = {"status": "ok"}
    if option_buy_module is None:
        status["note"] = "option module not imported; check logs for import errors."
    else:
        status["module"] = option_buy_module.__name__
    return status


@app.get("/debug")
def debug_module() -> JSONResponse:
    """
    Debug endpoint - lists imported module, file, available attributes, and import errors.
    Use for debugging on Railway (will appear in logs and response).
    """
    result = {
        "imported_module": option_buy_module.__name__ if option_buy_module else None,
        "module_file": getattr(option_buy_module, "__file__", None) if option_buy_module else None,
        "available_attrs": sorted([n for n in dir(option_buy_module) if not n.startswith("_")]) if option_buy_module else [],
        "bound": {k: (v.__name__ if callable(v) else type(v).__name__ if v is not None else None) for k, v in _bound.items()},
        "import_errors_try": import_errors,
        "env": {
            "GS_CREDS_JSON_exists": bool(os.environ.get("GS_CREDS_JSON") or _bound.get("GS_CREDS_JSON")),
            "GOOGLE_SHEET_NAME": os.environ.get("GOOGLE_SHEET_NAME", None),
        },
    }
    return JSONResponse(status_code=200, content=result)


@app.get("/analyze")
def analyze(
    symbol: str = Query(..., description="Underlying symbol, e.g. HDFCBANK or NIFTY"),
    log_to_sheet: bool = Query(False, description="If true, log output to Google Sheet (requires creds)."),
) -> JSONResponse:
    """
    Analyze an underlying symbol and return candidate option buys.
    This function calls into the loaded option-buy module at runtime.
    """

    # Pre-flight checks
    if option_buy_module is None:
        msg = f"No option module imported. Tried: {PREFERRED_MODULES}. Check server logs."
        logger.error(msg)
        return JSONResponse(status_code=500, content={"error": msg, "import_errors": import_errors})

    # Ensure required callables are available
    missing = []
    for fn in ("fetch_option_chain_playwright", "analyze_option_chain", "select_candidate_strikes"):
        if not hasattr(option_buy_module, fn):
            missing.append(fn)
    if missing:
        avail = sorted([n for n in dir(option_buy_module) if not n.startswith("_")])
        msg = f"option module missing required functions: {missing}. Available: {avail}"
        logger.error(msg)
        return JSONResponse(status_code=500, content={"error": msg})

    # Bind callables/vars now (use module attributes directly so you get the actual module's values)
    fetch_fn = getattr(option_buy_module, "fetch_option_chain_playwright")
    analyze_fn = getattr(option_buy_module, "analyze_option_chain")
    select_fn = getattr(option_buy_module, "select_candidate_strikes")
    log_fn = getattr(option_buy_module, "log_to_google_sheet", None)

    # Vars: prefer module-level vars; fall back to environment or defaults
    module_symbol = getattr(option_buy_module, "SYMBOL", None)
    CAPITAL = getattr(option_buy_module, "CAPITAL", None) or float(os.environ.get("CAPITAL", 10000))
    RISK_PCT = getattr(option_buy_module, "RISK_PER_TRADE_PCT", None) or float(os.environ.get("RISK_PER_TRADE_PCT", 1.0))
    GS_CREDS = getattr(option_buy_module, "GS_CREDS_JSON", None) or os.environ.get("GS_CREDS_JSON")
    SHEET_NAME = getattr(option_buy_module, "GOOGLE_SHEET_NAME", None) or os.environ.get("GOOGLE_SHEET_NAME", "OptionBuyLog")

    # preserve original module SYMBOL if present; set temporarily for the run to keep CLI behavior consistent
    original_symbol_value = None
    if hasattr(option_buy_module, "SYMBOL"):
        original_symbol_value = getattr(option_buy_module, "SYMBOL")
        setattr(option_buy_module, "SYMBOL", symbol.upper())

    try:
        logger.info("Starting fetch for symbol=%s", symbol.upper())
        # fetch option chain using module's fetch function
        data = fetch_fn(symbol.upper())
        if not data:
            msg = "No data returned from fetch function."
            logger.warning(msg)
            return JSONResponse(
                status_code=200,
                content={"symbol": symbol.upper(), "candidates": [], "message": msg},
            )

        # analyze chain into dataframe
        df = analyze_fn(data, symbol.upper())
        if df is None or getattr(df, "empty", False):
            msg = "No valid option data parsed."
            logger.info(msg)
            return JSONResponse(
                status_code=200,
                content={"symbol": symbol.upper(), "candidates": [], "message": msg},
            )

        # select candidates
        candidates = select_fn(df, CAPITAL, RISK_PCT)
        if not candidates:
            msg = "No candidates found after filtering. Try relaxing MIN_OI or MIN_VOLUME."
            logger.info(msg)
            return JSONResponse(
                status_code=200,
                content={"symbol": symbol.upper(), "candidates": [], "message": msg},
            )

        # Optional Google Sheets logging (best-effort)
        sheet_warning = None
        if log_to_sheet:
            try:
                # only try if a log function exists
                if log_fn and (GS_CREDS or GS_CREDS is not None):
                    log_fn(GS_CREDS, SHEET_NAME, candidates)
                    logger.info("Logged %d candidates to Google Sheet '%s' (sheet creds=%s present=%s).",
                                len(candidates), SHEET_NAME, GS_CREDS, os.path.exists(GS_CREDS) if GS_CREDS else False)
                else:
                    sheet_warning = "Google Sheets logging requested but log function or credentials missing."
                    logger.warning(sheet_warning)
            except Exception as e:
                logger.exception("Google Sheets logging failed: %s", e)
                sheet_warning = f"Failed to log to Google Sheet: {e}"

        # Success
        resp = {"symbol": symbol.upper(), "candidates": candidates}
        if sheet_warning:
            resp["warning"] = sheet_warning

        logger.info("Analyze finished for %s, found %d candidates", symbol.upper(), len(candidates))
        return JSONResponse(status_code=200, content=resp)

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Unhandled exception in /analyze: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb},
        )
    finally:
        # restore original module symbol to avoid side-effects
        try:
            if hasattr(option_buy_module, "SYMBOL"):
                setattr(option_buy_module, "SYMBOL", original_symbol_value)
        except Exception:
            logger.exception("Failed to restore original SYMBOL on option module.")
