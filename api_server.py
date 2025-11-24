# api_server.py
from fastapi import FastAPI, Query, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import traceback
import time
from typing import Optional

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

# -------- Logging configuration --------
LOGFILE = os.getenv("APP_LOGFILE", "/tmp/app.log")
LOGLEVEL = os.getenv("APP_LOGLEVEL", "INFO").upper()

logger = logging.getLogger("optionbuy_api")
logger.setLevel(getattr(logging, LOGLEVEL, logging.INFO))

# Stream handler -> stdout (Railway reads stdout)
sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(getattr(logging, LOGLEVEL, logging.INFO))
sh_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s'
)
sh.setFormatter(sh_formatter)
logger.addHandler(sh)

# Rotating file handler -> /tmp/app.log (inspect via /logs endpoint)
fh = RotatingFileHandler(LOGFILE, maxBytes=5_000_000, backupCount=5)
fh.setLevel(getattr(logging, LOGLEVEL, logging.INFO))
fh_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s'
)
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

# reduce verbosity of noisy libraries if needed
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

# -------- FastAPI app --------
app = FastAPI(title="Option Buy API", version="1.0.0")

# optional CORS (if you call from browser directly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Middleware to log requests and responses --------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        body = await request.body()
        logger.info(f"req.start path={request.url.path} method={request.method} q={request.url.query} client={request.client} body_len={len(body)}")
    except Exception:
        # some bodies are non-readable; ignore
        logger.debug("req.start unable to read body")

    try:
        response: Response = await call_next(request)
    except Exception as e:
        # Log exception with stack trace and re-raise so handler processes it
        tb = traceback.format_exc()
        logger.error(f"req.exception path={request.url.path} error={e}\n{tb}")
        raise

    process_time = (time.time() - start_time) * 1000.0
    logger.info(f"req.done path={request.url.path} status={response.status_code} time_ms={process_time:.1f}")
    return response


# -------- Startup / Shutdown events --------
@app.on_event("startup")
async def startup_event():
    logger.info("startup: OptionBuy API starting")
    # show some diagnostic info (playwright installed? creds existence)
    try:
        import importlib.util
        has_playwright = importlib.util.find_spec("playwright") is not None
    except Exception:
        has_playwright = False
    logger.info(f"startup: playwright_installed={has_playwright}")
    logger.info(f"startup: GS_CREDS_JSON_exists={os.path.exists(GS_CREDS_JSON) if GS_CREDS_JSON else False}")
    logger.info(f"startup: LOGFILE={LOGFILE}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("shutdown: OptionBuy API shutting down")


# -------- Exception handler so we capture stack traces cleanly --------
@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.exception(f"Unhandled exception for request {request.method} {request.url.path}: {exc}\n{tb}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# -------- Health endpoint --------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------- Tail logs endpoint --------
@app.get("/logs", response_class=PlainTextResponse)
def read_logs(lines: Optional[int] = 500):
    """
    Return the last `lines` lines from the app logfile (LOGFILE).
    Example: /logs?lines=200
    """
    try:
        n = int(lines) if lines else 500
    except Exception:
        n = 500

    if not os.path.exists(LOGFILE):
        return PlainTextResponse(f"No logfile found at {LOGFILE}", status_code=404)

    # Read last N lines efficiently
    def tail(path, n_lines):
        with open(path, "rb") as f:
            # Seek from end
            approx_chunk = 1024
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            data = bytearray()
            block_end = file_size
            while block_end > 0 and len(data.splitlines()) <= n_lines:
                block_start = max(0, block_end - approx_chunk)
                f.seek(block_start)
                chunk = f.read(block_end - block_start)
                data = chunk + data
                block_end = block_start
                if block_start == 0:
                    break
            lines_out = data.splitlines()[-n_lines:]
            return "\n".join([line.decode(errors="replace") for line in lines_out])

    try:
        out = tail(LOGFILE, n)
        return PlainTextResponse(out, status_code=200)
    except Exception as e:
        logger.exception("Failed to read logfile")
        return PlainTextResponse(f"Failed to read logfile: {e}", status_code=500)


# -------- Analyze endpoint (wrapped with robust logging) --------
@app.get("/analyze")
def analyze(
    symbol: str = Query(..., description="Underlying symbol, e.g. HDFCBANK or NIFTY"),
    log_to_sheet: bool = Query(
        False,
        description="If true, log output to Google Sheet (requires valid creds.json in container).",
    ),
):
    request_id = f"{int(time.time()*1000)}"
    logger.info(f"analyze.start id={request_id} symbol={symbol} log_to_sheet={log_to_sheet}")

    # Temporarily override SYMBOL for this request
    original_symbol = option_buy.SYMBOL
    option_buy.SYMBOL = symbol.upper()

    try:
        data = fetch_option_chain_playwright(option_buy.SYMBOL)
        logger.info(f"analyze: fetched option chain for {option_buy.SYMBOL} keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        df = analyze_option_chain(data, option_buy.SYMBOL)
        if df.empty:
            logger.warning(f"analyze.empty_data symbol={option_buy.SYMBOL}")
            return JSONResponse(
                status_code=200,
                content={
                    "symbol": option_buy.SYMBOL,
                    "candidates": [],
                    "message": "No valid option data parsed.",
                },
            )

        candidates = select_candidate_strikes(df, CAPITAL, RISK_PER_TRADE_PCT)
        logger.info(f"analyze: candidates_count={len(candidates)} for {option_buy.SYMBOL}")

        if not candidates:
            logger.info("analyze: no candidates after filtering")
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
                logger.info("analyze: logged to Google Sheet")
            except Exception as e:
                logger.exception("analyze: failed to log to Google Sheet")
                # return response that indicates logging failed but includes candidates
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
        # Already logged by middleware/exception handler, but include an informative response
        logger.exception(f"analyze.error id={request_id} symbol={symbol} error={e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # restore original SYMBOL so CLI behavior stays unchanged
        option_buy.SYMBOL = original_symbol
        logger.info(f"analyze.end id={request_id} symbol={symbol}")
