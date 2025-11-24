#!/usr/bin/env python3
"""
option_buy_api.py
API-friendly version of your NSE Option Picker logic.
- Playwright import is lazy (imported only when fetch_playwright is called).
- Has requests fallback (uses plain requests if Playwright fails).
- Exposes functions:
    fetch_option_chain(symbol)
    analyze_option_chain(data, symbol)
    select_candidate_strikes(df, capital, risk_pct)
    log_to_google_sheet(creds_json, sheet_name, rows)
- Keeps lot-size map, feature toggles, and reason text.
"""

import math
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# requests is used as a fallback if Playwright isn't available or fails.
import requests

# Google Sheets
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    GS_IMPORTED = True
except Exception:
    GS_IMPORTED = False

# ----------------------------- CONFIG -----------------------------
SYMBOL = os.getenv('SYMBOL', 'HDFCBANK')
CAPITAL = float(os.getenv('CAPITAL', 10000))
RISK_PER_TRADE_PCT = float(os.getenv('RISK_PER_TRADE_PCT', 1.0))
GOOGLE_SHEET_NAME = os.getenv('GOOGLE_SHEET_NAME', 'OptionBuyLog')
GS_CREDS_JSON = os.getenv('GS_CREDS_JSON', 'creds.json')

MAX_CANDIDATES = int(os.getenv('MAX_CANDIDATES', 1))
FORCE_ONE_OF_EACH = os.getenv('FORCE_ONE_OF_EACH', 'False').lower() in ('1','true','yes')
SELECT_EXPIRY = os.getenv('SELECT_EXPIRY', None)
MARKET_HOLIDAYS = os.getenv('MARKET_HOLIDAYS', '2025-01-26,2025-08-15').split(',')

MIN_OI = int(os.getenv('MIN_OI', 500))
MIN_VOLUME = int(os.getenv('MIN_VOLUME', 50))
TARGET_RR = float(os.getenv('TARGET_RR', 4.0))

# Lot sizes (keep updated)
LOTS_BY_SYMBOL = {
    "NIFTY": 75,
    "BANKNIFTY": 35,
    "FINNIFTY": 40,
    "MIDCPNIFTY": 140,
    "NIFTYNXT50": 25,
    "RELIANCE": 500,
    "HDFCBANK": 550,
    "TCS": 175,
    "INFY": 400,
    "ICICIBANK": 700,
    "AXISBANK": 625,
    "SBIN": 750,
    "KOTAKBANK": 400,
    "LT": 175,
}

GLOBAL_LOT_SIZE: Optional[int] = None
ROUND_TO_LOTS = True
MIN_LOTS = 1

# Feature toggles and weights (you can change these)
FACTOR_ENABLE = {
    'oi': True,
    'oi_change': True,
    'volume': True,
    'atm_proximity': False,
    'prob_itm': False,
    'directional_oi': True,
}

FACTOR_WEIGHTS = {
    'oi': 0.30,
    'oi_change': 0.25,
    'volume': 0.15,
    'atm_proximity': 0.20,
    'prob_itm': 0.10,
    'directional_oi': 0.20,
}

SYMBOL_LOT_ALIASES = {
    "NIFTY50": "NIFTY",
    "NIFTY 50": "NIFTY",
    "BANK NIFTY": "BANKNIFTY",
    "NIFTYBANK": "BANKNIFTY",
}

# ----------------------------- UTIL -----------------------------
def is_market_day(today: datetime) -> bool:
    if today.weekday() >= 5:
        return False
    if today.strftime('%Y-%m-%d') in MARKET_HOLIDAYS:
        return False
    return True

def get_lot_size_for_symbol(sym: str) -> int:
    if GLOBAL_LOT_SIZE and isinstance(GLOBAL_LOT_SIZE, int) and GLOBAL_LOT_SIZE > 0:
        return GLOBAL_LOT_SIZE
    if not sym:
        return 1
    key = str(sym).upper().strip()
    if key in LOTS_BY_SYMBOL:
        return int(LOTS_BY_SYMBOL[key])
    if key in SYMBOL_LOT_ALIASES:
        alias = SYMBOL_LOT_ALIASES[key]
        if alias in LOTS_BY_SYMBOL:
            return int(LOTS_BY_SYMBOL[alias])
    for suffix in (".NS", " EQ", "EQ"):
        if key.endswith(suffix):
            k2 = key.replace(suffix, "").strip()
            if k2 in LOTS_BY_SYMBOL:
                return int(LOTS_BY_SYMBOL[k2])
    k_alpha = "".join(ch for ch in key if not ch.isdigit()).strip()
    if k_alpha in LOTS_BY_SYMBOL:
        return int(LOTS_BY_SYMBOL[k_alpha])
    # fallback
    return 1

# ----------------------------- FETCHERS (Playwright lazy + requests fallback) -----------------------------
NSE_HOME = "https://www.nseindia.com"
OC_IDX = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
OC_EQU = "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
INDEX_SYMBOLS = {"NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"}

UA_FALLBACK = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/127.0.0.0 Safari/537.36")

def _requests_get_chain(symbol: str, timeout: int = 20) -> Optional[dict]:
    url = OC_IDX if symbol.upper() in INDEX_SYMBOLS else OC_EQU
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA_FALLBACK,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/option-chain",
    })
    try:
        # friendly pre-requests
        s.get(NSE_HOME, timeout=10)
        s.get("https://www.nseindia.com/option-chain", timeout=10)
    except Exception:
        pass
    try:
        r = s.get(url.format(symbol=symbol.upper()), timeout=timeout)
        ctype = (r.headers.get("Content-Type") or "").lower()
        txt = r.text or ""
        if r.status_code == 200:
            if "application/json" in ctype:
                return r.json()
            if txt and not txt.lstrip().startswith("<"):
                return json.loads(txt)
    except Exception:
        pass
    return None

def _playwright_get_chain(symbol: str, timeout: int = 60) -> Optional[dict]:
    # lazy import: only import Playwright if available and needed
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None
    url = OC_IDX if symbol.upper() in INDEX_SYMBOLS else OC_EQU
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=UA_FALLBACK,
                                          extra_http_headers={"Accept": "application/json, text/plain, */*", "Referer": "https://www.nseindia.com/option-chain"},
                                          viewport={"width":1366,"height":768},
                                          ignore_https_errors=True)
            page = context.new_page()
            try:
                page.goto(NSE_HOME, timeout=30000)
                page.wait_for_timeout(700)
            except Exception:
                pass
            try:
                page.goto("https://www.nseindia.com/option-chain", timeout=30000)
                page.wait_for_timeout(1100)
            except Exception:
                pass
            try:
                resp = context.request.get(url.format(symbol=symbol.upper()), timeout=timeout*1000)
                ctype = (resp.headers.get("content-type") or "").lower()
                txt = resp.text()
                if resp.ok and "application/json" in ctype:
                    return resp.json()
                if resp.ok and txt and not txt.lstrip().startswith("<"):
                    return json.loads(txt)
            except Exception:
                pass
            try:
                context.storage_state(path=".nse_storage_state.json")
            except Exception:
                pass
            context.close()
            browser.close()
    except Exception:
        return None
    return None

def fetch_option_chain(symbol: str) -> dict:
    """
    Tries requests first (fast), then Playwright fallback.
    Raises RuntimeError if both fail.
    """
    # 1) try requests
    data = _requests_get_chain(symbol)
    if data:
        return data
    # 2) try playwright (if installed)
    data = _playwright_get_chain(symbol)
    if data:
        return data
    raise RuntimeError("Failed to fetch option chain (requests + playwright both failed).")

# ----------------------------- MATH / MODEL -----------------------------
def bs_prob_finish_itm(S: float, K: float, r: float, T: float, sigma: float, option_type: str) -> float:
    if sigma <= 0 or T <= 0:
        return 0.0
    try:
        d2 = (math.log(S / K) + (0.0 - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    except Exception:
        return 0.0
    if option_type.upper() == 'CE':
        return float(norm.cdf(d2))
    else:
        return float(1 - norm.cdf(d2))

# ----------------------------- ANALYSIS / SCORING -----------------------------
def analyze_option_chain(data: Dict, symbol: str) -> pd.DataFrame:
    records = []
    underlying = data.get('records', {}).get('underlyingValue') or data.get('records', {}).get('underlying')
    expiries = data.get('records', {}).get('expiryDates', [])
    options = data.get('records', {}).get('data', [])
    for row in options:
        strike = float(row.get('strikePrice') or 0)
        expiry = row.get('expiryDate') or row.get('expiry') or (expiries[0] if expiries else None)
        ce = row.get('CE')
        pe = row.get('PE')
        if ce:
            records.append({
                'strike': strike, 'type': 'CE', 'ltp': ce.get('lastPrice') or np.nan,
                'oi': ce.get('openInterest') or 0, 'oi_change': ce.get('changeinOpenInterest') or 0,
                'vol': ce.get('totalTradedVolume') or 0, 'iv': ce.get('impliedVolatility') or np.nan,
                'delta': ce.get('delta') or np.nan, 'expiry': expiry
            })
        if pe:
            records.append({
                'strike': strike, 'type': 'PE', 'ltp': pe.get('lastPrice') or np.nan,
                'oi': pe.get('openInterest') or 0, 'oi_change': pe.get('changeinOpenInterest') or 0,
                'vol': pe.get('totalTradedVolume') or 0, 'iv': pe.get('impliedVolatility') or np.nan,
                'delta': pe.get('delta') or np.nan, 'expiry': expiry
            })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df['underlying'] = float(underlying or 0.0)
    df['atm_diff'] = abs(df['strike'] - df['underlying'])
    df['expiry_dt'] = pd.to_datetime(df['expiry'], errors='coerce', utc=True)
    return df

def select_candidate_strikes(df: pd.DataFrame, capital: float, risk_pct: float) -> List[Dict[str,Any]]:
    candidates = []
    if df.empty:
        return candidates

    # expiry selection
    if 'expiry_dt' in df.columns:
        df['expiry_dt'] = pd.to_datetime(df['expiry_dt'], errors='coerce', utc=True)
    if SELECT_EXPIRY:
        sel_dt = pd.to_datetime(SELECT_EXPIRY, errors='coerce', utc=True)
        if pd.notna(sel_dt):
            df = df[df['expiry_dt'].dt.date == sel_dt.date()].copy()
    else:
        if 'expiry_dt' in df.columns and df['expiry_dt'].notna().any():
            nearest = df['expiry_dt'].min()
            df = df[df['expiry_dt'] == nearest].copy()

    if df.empty:
        return candidates

    S = float(df['underlying'].iat[0])

    def norm_col(col):
        arr = np.array(col, dtype=float)
        if arr.size == 0 or arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    df = df.copy()
    df['oi_n'] = norm_col(df['oi'])
    df['oi_ch_n'] = norm_col(np.maximum(df['oi_change'], 0))
    df['vol_n'] = norm_col(df['vol'])
    df['atm_n'] = 1 - norm_col(df['atm_diff'])

    if 'expiry_dt' in df.columns and df['expiry_dt'].notna().any():
        today = pd.Timestamp.now(tz='UTC').normalize()
        try:
            df['expiry_dt'] = df['expiry_dt'].dt.tz_convert('UTC')
        except Exception:
            df['expiry_dt'] = pd.to_datetime(df['expiry_dt'], errors='coerce', utc=True)
        df['T_days'] = (df['expiry_dt'] - today).dt.days.clip(lower=0)
        df['T_days'] = df['T_days'].replace(0, 1)
        df['T'] = df['T_days'] / 365.0
    else:
        df['T_days'] = 7
        df['T'] = 7 / 365.0

    df['iv_f'] = df['iv'].fillna(0.25) / 100.0
    df['prob_itm'] = [bs_prob_finish_itm(S, r['strike'], 0.06, float(r['T']), r['iv_f'], r['type']) for _, r in df.iterrows()]

    # directional OI pattern
    call_doi_total = float(df[df['type']=='CE']['oi_change'].sum())
    put_doi_total = float(df[df['type']=='PE']['oi_change'].sum())
    if call_doi_total < 0 and put_doi_total > 0:
        dir_bias = "BULL_FOR_CE"
    elif call_doi_total > 0 and put_doi_total < 0:
        dir_bias = "BEAR_FOR_PE"
    elif (call_doi_total > 0 and put_doi_total > 0) or (call_doi_total < 0 and put_doi_total < 0):
        dir_bias = "SIDEWAYS"
    else:
        dir_bias = "NEUTRAL"

    df['dir_oi_score'] = 0.5
    if dir_bias == "BULL_FOR_CE":
        df.loc[df['type']=='CE','dir_oi_score'] = 1.0
        df.loc[df['type']=='PE','dir_oi_score'] = 0.0
    elif dir_bias == "BEAR_FOR_PE":
        df.loc[df['type']=='PE','dir_oi_score'] = 1.0
        df.loc[df['type']=='CE','dir_oi_score'] = 0.0
    elif dir_bias == "SIDEWAYS":
        df['dir_oi_score'] = 0.0

    # score composition using toggles
    comps = []
    if FACTOR_ENABLE.get('oi', False):
        comps.append(FACTOR_WEIGHTS.get('oi',0.0) * df['oi_n'])
    if FACTOR_ENABLE.get('oi_change', False):
        comps.append(FACTOR_WEIGHTS.get('oi_change',0.0) * df['oi_ch_n'])
    if FACTOR_ENABLE.get('volume', False):
        comps.append(FACTOR_WEIGHTS.get('volume',0.0) * df['vol_n'])
    if FACTOR_ENABLE.get('atm_proximity', False):
        comps.append(FACTOR_WEIGHTS.get('atm_proximity',0.0) * df['atm_n'])
    if FACTOR_ENABLE.get('prob_itm', False):
        comps.append(FACTOR_WEIGHTS.get('prob_itm',0.0) * df['prob_itm'])
    if FACTOR_ENABLE.get('directional_oi', False):
        comps.append(FACTOR_WEIGHTS.get('directional_oi',0.0) * df['dir_oi_score'])

    if comps:
        df['score'] = sum(comps)
    else:
        df['score'] = 0.0

    # filter low liquidity
    df = df[(df['oi'] >= MIN_OI) & (df['vol'] >= MIN_VOLUME)].copy()
    if df.empty:
        return candidates

    max_total = MAX_CANDIDATES if MAX_CANDIDATES and MAX_CANDIDATES>0 else 1
    if FORCE_ONE_OF_EACH:
        ce_df = df[df['type']=='CE'].sort_values(['score','prob_itm'], ascending=False)
        pe_df = df[df['type']=='PE'].sort_values(['score','prob_itm'], ascending=False)
        chosen = []
        if not ce_df.empty:
            chosen.append(ce_df.iloc[0])
        if not pe_df.empty:
            chosen.append(pe_df.iloc[0])
        rem = max_total - len(chosen)
        if rem>0:
            others = df.drop(index=[r.name for r in chosen], errors='ignore').sort_values(['score','prob_itm'], ascending=False).head(rem)
            chosen.extend([row for _,row in others.iterrows()])
        top_rows = pd.DataFrame(chosen) if chosen else pd.DataFrame(columns=df.columns)
    else:
        top_rows = df.sort_values(['score','prob_itm'], ascending=False).head(max_total)

    for _, r in top_rows.iterrows():
        try:
            ltp_val = float(r['ltp'])
            premium = max(0.5, ltp_val) if not math.isnan(ltp_val) and ltp_val>0 else 0.5
        except Exception:
            premium = 0.5

        stop_loss_price = premium * 0.5
        target_price = premium * (1 + TARGET_RR)
        risk_amount = (premium - stop_loss_price)
        allowed_risk = (capital * (risk_pct / 100.0))
        denom = risk_amount if (risk_amount and risk_amount>0) else premium
        qty_units = int(max(1, math.floor(allowed_risk / denom))) if denom>0 else 1

        lot_size = get_lot_size_for_symbol(SYMBOL)
        if ROUND_TO_LOTS:
            lots = max(MIN_LOTS, qty_units // lot_size)
            if lots == 0 and qty_units >= lot_size:
                lots = 1
            qty = int(lots * lot_size)
        else:
            qty = qty_units
            lots = math.ceil(qty_units / lot_size) if lot_size>0 else 0

        conviction = float(min(99.9, 100*float(r.get('score',0.0))))
        if dir_bias == "SIDEWAYS":
            conviction *= 0.5

        contribs = {}
        if FACTOR_ENABLE.get('oi', False):
            contribs['OI'] = FACTOR_WEIGHTS.get('oi',0.0) * float(r['oi_n'])
        if FACTOR_ENABLE.get('oi_change', False):
            contribs['OI_change'] = FACTOR_WEIGHTS.get('oi_change',0.0) * float(r['oi_ch_n'])
        if FACTOR_ENABLE.get('volume', False):
            contribs['Volume'] = FACTOR_WEIGHTS.get('volume',0.0) * float(r['vol_n'])
        if FACTOR_ENABLE.get('atm_proximity', False):
            contribs['ATM_proximity'] = FACTOR_WEIGHTS.get('atm_proximity',0.0) * float(r['atm_n'])
        if FACTOR_ENABLE.get('prob_itm', False):
            contribs['Prob_ITM'] = FACTOR_WEIGHTS.get('prob_itm',0.0) * float(r['prob_itm'])
        if FACTOR_ENABLE.get('directional_oi', False):
            contribs['Directional_OI'] = FACTOR_WEIGHTS.get('directional_oi',0.0) * float(r['dir_oi_score'])
        sorted_contrib = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        top_factors = ', '.join([f"{name}({value:.3f})" for name,value in sorted_contrib[:3]]) if sorted_contrib else 'No factors enabled'

        if dir_bias == "BULL_FOR_CE":
            dir_comment = "Call OI ↓ & Put OI ↑ → favourable for CE buying."
        elif dir_bias == "BEAR_FOR_PE":
            dir_comment = "Put OI ↓ & Call OI ↑ → favourable for PE buying."
        elif dir_bias == "SIDEWAYS":
            dir_comment = "Both Call & Put OI moving same direction → avoid aggressive option buying."
        else:
            dir_comment = "Neutral/mixed OI."

        reason_parts = [
            f"Chosen because top combined score among strikes for expiry {r.get('expiry')}.",
            f"Top factors: {top_factors}.",
            f"Underlying={S}, Strike={r['strike']}, LTP={premium}, IV={float(r['iv_f']*100):.2f}%, Prob_ITM={float(r['prob_itm']):.3%}.",
            f"OI={int(r['oi'])}, ΔOI={int(r['oi_change'])}, Volume={int(r['vol'])}, ATM_diff={abs(r['atm_diff']):.2f}.",
            f"Total ΔOI: Call ΔOI={call_doi_total:+.0f}, Put ΔOI={put_doi_total:+.0f}. {dir_comment}",
            f"Score={float(r['score']):.4f}, Conviction={conviction:.1f}%.",
            f"Trade sizing => Qty={qty} units ({lots} lots of size {lot_size}) for risk {risk_pct}% of capital ({allowed_risk}). Stop={round(stop_loss_price,2)}, Target={round(target_price,2)} (R:R={TARGET_RR}:1)."
        ]
        reason_text = ' '.join(reason_parts)

        candidates.append({
            'symbol': SYMBOL,
            'type': r['type'],
            'strike': r['strike'],
            'ltp': premium,
            'oi': int(r['oi']),
            'oi_change': int(r['oi_change']),
            'iv': float(r['iv_f']),
            'prob_itm': float(r['prob_itm']),
            'score': float(r['score']),
            'conviction_pct': conviction,
            'qty': qty,
            'lots': int(lots),
            'lot_size': int(lot_size),
            'stop_loss': round(stop_loss_price,2),
            'target': round(target_price,2),
            'rr': TARGET_RR,
            'expiry': str(r.get('expiry')),
            'T_days': int(r.get('T_days', -1)),
            'reason': reason_text
        })

    return candidates

# ----------------------------- GOOGLE SHEETS LOG -----------------------------
def log_to_google_sheet(creds_json: str, sheet_name: str, rows: List[Dict]):
    if not GS_IMPORTED:
        raise RuntimeError("gspread/oauth2client not installed in environment.")
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
    client = gspread.authorize(creds)
    try:
        sheet = client.open(sheet_name).sheet1
    except Exception:
        sh = client.create(sheet_name)
        sheet = sh.sheet1
    header = ['timestamp','symbol','type','strike','ltp','qty','lots','lot_size','oi','oi_change','iv','prob_itm','conviction_pct','stop_loss','target','rr','expiry','T_days','reason']
    try:
        existing = sheet.row_values(1)
        if not existing or existing[0].lower() != 'timestamp':
            sheet.insert_row(header,1)
    except Exception:
        pass
    for r in rows:
        row = [datetime.utcnow().isoformat(), r['symbol'], r['type'], r['strike'], r['ltp'], r['qty'], r.get('lots'), r.get('lot_size'), r['oi'], r['oi_change'], r['iv'], r['prob_itm'], r['conviction_pct'], r['stop_loss'], r['target'], r['rr'], r['expiry'], r['T_days'], r['reason']]
        sheet.append_row(row)

# Expose module-level helper names used by api_server.py
__all__ = [
    "fetch_option_chain",
    "analyze_option_chain",
    "select_candidate_strikes",
    "log_to_google_sheet",
    "SYMBOL",
    "CAPITAL",
    "RISK_PER_TRADE_PCT",
    "GOOGLE_SHEET_NAME",
    "GS_CREDS_JSON",
]
