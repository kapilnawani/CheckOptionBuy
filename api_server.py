# api_server.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import os

import option_buy
from option_buy import (
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

app = FastAPI(title="Option Buy API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/analyze")
def analyze(
    symbol: str = Query(..., description="Underlying symbol, e.g. HDFCBANK or NIFTY"),
    log_to_sheet: bool = Query(
        False,
        description="If true, log output to Google Sheet (requires valid creds.json in container).",
    ),
):
    # Temporarily override SYMBOL for this request
    original_symbol = option_buy.SYMBOL
    option_buy.SYMBOL = symbol.upper()

    try:
        data = fetch_option_chain_playwright(option_buy.SYMBOL)
        df = analyze_option_chain(data, option_buy.SYMBOL)
        if df.empty:
            return JSONResponse(
                status_code=200,
                content={
                    "symbol": option_buy.SYMBOL,
                    "candidates": [],
                    "message": "No valid option data parsed.",
                },
            )

        candidates = select_candidate_strikes(df, CAPITAL, RISK_PER_TRADE_PCT)
        if not candidates:
            return JSONResponse(
                status_code=200,
                content={
                    "symbol": option_buy.SYMBOL,
                    "candidates": [],
                    "message": "No candidates found after filtering. Try relaxing MIN_OI or MIN_VOLUME.",
                },
            )

        # Optional Google Sheets logging
        if log_to_sheet and GS_CREDS_JSON and os.path.exists(GS_CREDS_JSON):
            try:
                log_to_google_sheet(GS_CREDS_JSON, GOOGLE_SHEET_NAME, candidates)
            except Exception as e:
                # Don't kill the API if logging fails; just report it in response
                return JSONResponse(
                    status_code=200,
                    content={
                        "symbol": option_buy.SYMBOL,
                        "candidates": candidates,
                        "warning": f"Failed to log to Google Sheet: {e}",
                    },
                )

        return JSONResponse(
            status_code=200,
            content={
                "symbol": option_buy.SYMBOL,
                "candidates": candidates,
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        # restore original SYMBOL so CLI behavior stays unchanged
        option_buy.SYMBOL = original_symbol
