# api_server.py (replace your current file)
import os
import traceback
import logging
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# configure logging to stdout (Railway will capture this)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


import option_buy_api as option_buy
from option_buy_api import (
    fetch_option_chain_playwright,
    analyze_option_chain,
    select_candidate_strikes,
    log_to_google_sheet,
    SYMBOL,
    CAPITAL,
    RISK_PER_TRADE_PCT,
    GOOGLE_SHEET_NAME,
    GS_CREDS_JSON,
)

app = FastAPI(title="Option Buy API", version="1.0.1")

# quick health
@app.get("/health")
def health():
    return {"status": "ok"}

# debug/diagnostics endpoint
@app.get("/debug")
def debug():
    info = {
        "symbol_default": option_buy.SYMBOL,
        "min_oi": option_buy.MIN_OI,
        "min_volume": option_buy.MIN_VOLUME,
        "max_candidates": option_buy.MAX_CANDIDATES,
        "feature_toggles": getattr(option_buy, "FACTOR_ENABLE", {}),
        "lots_by_symbol_sample": dict(list(getattr(option_buy, "LOTS_BY_SYMBOL", {}).items())[:10]),
    }
    return JSONResponse(status_code=200, content=info)

@app.get("/analyze")
def analyze(
    symbol: str = Query(..., description="Underlying symbol, e.g. HDFCBANK or NIFTY"),
    log_to_sheet: bool = Query(False, description="If true, log output to Google Sheet (requires creds.json)"),
):
    logging.info(f"/analyze called symbol={symbol} log_to_sheet={log_to_sheet}")
    original_symbol = option_buy.SYMBOL
    option_buy.SYMBOL = symbol.upper()

    try:
        # Use unified fetcher function name used in patch A
        logging.info("fetching chain for %s ...", option_buy.SYMBOL)
        data = option_buy.fetch_chain(option_buy.SYMBOL)
        logging.info("fetched chain, parsing...")
        df = analyze_option_chain(data, option_buy.SYMBOL)
        if df.empty:
            return JSONResponse(status_code=200, content={"symbol": option_buy.SYMBOL, "candidates": [], "message": "No valid option data parsed."})

        candidates = select_candidate_strikes(df, option_buy.CAPITAL, option_buy.RISK_PER_TRADE_PCT)
        if not candidates:
            return JSONResponse(status_code=200, content={"symbol": option_buy.SYMBOL, "candidates": [], "message": "No candidates found after filtering. Try relaxing MIN_OI or MIN_VOLUME."})

        # Optional logging to Google Sheets but don't crash on failure
        if log_to_sheet and option_buy.GS_CREDS_JSON and os.path.exists(option_buy.GS_CREDS_JSON):
            try:
                log_to_google_sheet(option_buy.GS_CREDS_JSON, option_buy.GOOGLE_SHEET_NAME, candidates)
            except Exception as e:
                logging.exception("Google Sheets logging failed")
                return JSONResponse(status_code=200, content={"symbol": option_buy.SYMBOL, "candidates": candidates, "warning": f"Failed to log to Google Sheet: {e}"})

        return JSONResponse(status_code=200, content={"symbol": option_buy.SYMBOL, "candidates": candidates})

    except Exception as e:
        # log full traceback to stdout so Railway shows it in logs
        logging.exception("Unhandled exception in /analyze")
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})
    finally:
        option_buy.SYMBOL = original_symbol
