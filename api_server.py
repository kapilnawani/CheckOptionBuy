# api_server.py
import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import Optional

# import the API-friendly option module (lazy import of playwright happens inside)
import option_buy_api as ob

# Configure structured logging to a file (JSON lines) so Railway logs show details on crashes
LOG_PATH = os.getenv("APP_LOG_PATH", "app_logs.jsonl")
logger = logging.getLogger("option_buy_api")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(title="Option Buy API", version="1.0.0")

def _log_event(event: str, **fields):
    payload = {"ts": datetime.utcnow().isoformat()+"Z", "event": event}
    payload.update(fields)
    try:
        logger.info(json.dumps(payload))
    except Exception:
        # fallback to stdout for Railway
        print(json.dumps(payload))

@app.get("/health")
def health():
    _log_event("health.check", status="ok")
    return {"status":"ok"}

@app.get("/analyze")
def analyze(symbol: str = Query(..., description="Underlying symbol, e.g. HDFCBANK or NIFTY"),
            log_to_sheet: bool = Query(False, description="Log to Google Sheet if credentials available.")):
    symbol = symbol.upper().strip()
    _log_event("request.start", symbol=symbol, log_to_sheet=log_to_sheet)
    # temporarily set module SYMBOL for behavior that relies on it (kept minimal)
    original_sym = ob.SYMBOL
    ob.SYMBOL = symbol
    try:
        raw = ob.fetch_option_chain(symbol)
        df = ob.analyze_option_chain(raw, symbol)
        if df.empty:
            _log_event("no_data", symbol=symbol)
            return JSONResponse(status_code=200, content={"symbol":symbol, "candidates":[], "message":"No valid option data parsed."})
        candidates = ob.select_candidate_strikes(df, ob.CAPITAL, ob.RISK_PER_TRADE_PCT)
        if not candidates:
            _log_event("no_candidates", symbol=symbol)
            return JSONResponse(status_code=200, content={"symbol":symbol, "candidates":[], "message":"No candidates after filtering (try relax MIN_OI/MIN_VOLUME)."})
        # optional sheet logging
        sheet_warning = None
        if log_to_sheet and ob.GS_CREDS_JSON and os.path.exists(ob.GS_CREDS_JSON):
            try:
                ob.log_to_google_sheet(ob.GS_CREDS_JSON, ob.GOOGLE_SHEET_NAME, candidates)
            except Exception as e:
                sheet_warning = str(e)
                _log_event("sheet.log_failed", error=sheet_warning)
        _log_event("request.success", symbol=symbol, num_candidates=len(candidates))
        resp = {"symbol":symbol, "candidates": candidates}
        if sheet_warning:
            resp["sheet_warning"] = sheet_warning
        return JSONResponse(status_code=200, content=resp)
    except Exception as e:
        err_str = str(e)
        _log_event("request.error", symbol=symbol, error=err_str)
        return JSONResponse(status_code=500, content={"error": err_str})
    finally:
        ob.SYMBOL = original_sym
        _log_event("request.end", symbol=symbol)
