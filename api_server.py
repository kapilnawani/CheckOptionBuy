# api_server.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# Import your existing logic
import option_buy
from option_buy import (
    fetch_option_chain_playwright,
    analyze_option_chain,
    select_candidate_strikes,
)

app = FastAPI(title="Option Buy API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/analyze")
def analyze(
    symbol: str = Query("HDFCBANK", description="Underlying symbol, e.g. HDFCBANK, NIFTY"),
    capital: float = Query(option_buy.CAPITAL, description="Total capital in INR"),
    risk_pct: float = Query(option_buy.RISK_PER_TRADE_PCT, description="Risk per trade in %"),
    max_candidates: int = Query(option_buy.MAX_CANDIDATES, description="How many suggestions to return"),
):
    """
    Run your existing engine and return the candidate list as JSON.
    """
    try:
        sym = symbol.upper().strip()

        # Override globals in option_buy for this request (so candidates['symbol'] is correct)
        option_buy.SYMBOL = sym
        option_buy.CAPITAL = capital
        option_buy.RISK_PER_TRADE_PCT = risk_pct
        option_buy.MAX_CANDIDATES = max_candidates

        data = fetch_option_chain_playwright(sym)
        df = analyze_option_chain(data, sym)

        if df.empty:
            return JSONResponse(
                status_code=404,
                content={"error": "No valid option data parsed for symbol", "symbol": sym},
            )

        candidates = select_candidate_strikes(df, capital, risk_pct)

        if not candidates:
            return JSONResponse(
                status_code=404,
                content={"error": "No candidates found after filtering", "symbol": sym},
            )

        return {
            "symbol": sym,
            "capital": capital,
            "risk_pct": risk_pct,
            "max_candidates": max_candidates,
            "count": len(candidates),
            "candidates": candidates,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})