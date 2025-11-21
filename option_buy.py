#!/usr/bin/env python3
"""
NSE Option Picker
Single-file Python script that uses Playwright to fetch the NSE option-chain JSON, analyzes OI & Greeks,
and suggests option-buy trades (CE/PE) with scoring, probability and trade instructions.

Requirements (install):
  pip install playwright pandas numpy scipy gspread oauth2client python-dotenv
  playwright install

Run:
  python nse_option_picker.py
"""
import math
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy.stats import norm

# Playwright imports
from playwright.sync_api import sync_playwright

# Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ----------------------------- CONFIG -----------------------------
SYMBOL = os.getenv('SYMBOL', 'HDFCBANK')
#SYMBOL = os.getenv('SYMBOL', 'SBIN')
CAPITAL = float(os.getenv('CAPITAL', 10000))
RISK_PER_TRADE_PCT = float(os.getenv('RISK_PER_TRADE_PCT', 1.0))
GOOGLE_SHEET_NAME = os.getenv('GOOGLE_SHEET_NAME', 'OptionBuyLog')
GS_CREDS_JSON = os.getenv('GS_CREDS_JSON', 'creds.json')

# MAX_CANDIDATES controls the TOTAL number of suggestions returned.
MAX_CANDIDATES = 1  # how many total CE/PE suggestions to return
# If True, force one CE + one PE (if available). If MAX_CANDIDATES>2, the remainder
# will be filled by highest-score options regardless of side.
FORCE_ONE_OF_EACH = False
# Optionally select a specific expiry (string like '2025-11-20' or '20-Nov-2025')
# If None, the script will choose the nearest expiry automatically.
SELECT_EXPIRY = None  # e.g. '2025-11-20' or None

MARKET_HOLIDAYS = os.getenv('MARKET_HOLIDAYS', '2025-01-26,2025-08-15').split(',')

# Strategy hyperparams (tweak as you test)
MIN_OI = 500       # ignore strikes with OI less than this
MIN_VOLUME = 50    # ignore low volume
TARGET_RR = 4.0    # required R:R

# Lot-sizing: configure per-symbol lot sizes here. VERIFY these values for your instruments.
LOTS_BY_SYMBOL = {
    # Indices
    "NIFTY": 75,
    "BANKNIFTY": 35,
    "FINNIFTY": 40,
    "MIDCPNIFTY": 140,
    "NIFTYNXT50": 25,

    # Popular stock futures/options (common F&O names)
    "RELIANCE": 500,
    "HDFCBANK": 550,
    "TCS": 175,
    "INFY": 400,
    "ICICIBANK": 700,
    "AXISBANK": 625,
    "SBIN": 750,
    "KOTAKBANK": 400,
    "LT": 175,
    # add other symbols you trade here...
}
GLOBAL_LOT_SIZE = None  # override for all symbols if set
ROUND_TO_LOTS = True
MIN_LOTS = 1  # minimum lots to suggest

# ----------------------------- FEATURE TOGGLES -----------------------------
# Enable/disable scoring factors here
FACTOR_ENABLE = {
    'oi': True,
    'oi_change': True,
    'volume': True,
    'atm_proximity': False,
    'prob_itm': False,
    'directional_oi': True,   # NEW: based on total Call/Put OI change pattern
}

# Configurable weights for factors (sum not required; any zeroed factor won't be used)
FACTOR_WEIGHTS = {
    'oi': 0.30,
    'oi_change': 0.25,
    'volume': 0.15,
    'atm_proximity': 0.20,
    'prob_itm': 0.10,
    'directional_oi': 0.20,   # NEW: weight for directional OI bias
}

# ----------------------------- LOT-SIZE HELPERS -----------------------------
SYMBOL_LOT_ALIASES = {
    "NIFTY50": "NIFTY",
    "NIFTY 50": "NIFTY",
    "BANK NIFTY": "BANKNIFTY",
    "NIFTYBANK": "BANKNIFTY",
    # add other aliases if needed
}

def get_lot_size_for_symbol(sym: str) -> int:
    """
    Return lot size (int) for the given underlying symbol.
    Lookup order:
      1) GLOBAL_LOT_SIZE override
      2) exact key in LOTS_BY_SYMBOL (case-insensitive)
      3) alias mapping lookup
      4) suffix-normalizations like '.NS' or ' EQ'
      5) fallback to 1 (and print a warning)
    """
    if GLOBAL_LOT_SIZE and isinstance(GLOBAL_LOT_SIZE, int) and GLOBAL_LOT_SIZE > 0:
        return GLOBAL_LOT_SIZE

    if not sym:
        return 1

    key = str(sym).upper().strip()
    # direct lookup
    if key in LOTS_BY_SYMBOL:
        return int(LOTS_BY_SYMBOL[key])

    # alias
    if key in SYMBOL_LOT_ALIASES:
        alias = SYMBOL_LOT_ALIASES[key]
        if alias in LOTS_BY_SYMBOL:
            return int(LOTS_BY_SYMBOL[alias])

    # suffix normalization
    for suffix in (".NS", " EQ", "EQ"):
        if key.endswith(suffix):
            k2 = key.replace(suffix, "").strip()
            if k2 in LOTS_BY_SYMBOL:
                return int(LOTS_BY_SYMBOL[k2])

    # strip digits/extra
    k_alpha = "".join(ch for ch in key if not ch.isdigit()).strip()
    if k_alpha in LOTS_BY_SYMBOL:
        return int(LOTS_BY_SYMBOL[k_alpha])

    print(f"[WARN] Lot-size for symbol '{sym}' not found in LOTS_BY_SYMBOL. Defaulting to 1.")
    return 1

# ----------------------------- UTIL -----------------------------
def is_market_day(today: datetime) -> bool:
    # Basic check for weekend + custom holidays
    if today.weekday() >= 5:
        return False
    if today.strftime('%Y-%m-%d') in MARKET_HOLIDAYS:
        return False
    return True

# ----------------------------- FETCHING DATA -----------------------------
def fetch_option_chain_playwright(symbol: str) -> Dict:
    """Robust Playwright fetcher for NSE option-chain JSON."""
    api_idx = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    api_equ = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    url = api_idx if symbol.upper() in {"NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"} else api_equ

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context_kwargs = {
            "user_agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/127.0.0.0 Safari/537.36"),
            "extra_http_headers": {"Accept": "application/json, text/plain, */*", "Referer": "https://www.nseindia.com/option-chain"},
            "viewport": {"width": 1366, "height": 768},
            "locale": "en-US",
            "ignore_https_errors": True,
        }
        storage_state_path = ".nse_storage_state.json"
        if os.path.exists(storage_state_path):
            context_kwargs["storage_state"] = storage_state_path

        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        try:
            page.goto('https://www.nseindia.com', timeout=30000)
            page.wait_for_timeout(800)
        except Exception:
            pass
        try:
            page.goto('https://www.nseindia.com/option-chain', timeout=30000)
            page.wait_for_timeout(1200)
        except Exception:
            pass

        try:
            context.storage_state(path=storage_state_path)
        except Exception:
            pass

        data = None
        try:
            resp = context.request.get(url, timeout=60000)
            ctype = (resp.headers.get('content-type') or '').lower()
            text = resp.text()
            if resp.ok and 'application/json' in ctype:
                data = resp.json()
            elif resp.ok and text and not text.lstrip().startswith('<'):
                try:
                    data = json.loads(text)
                except Exception:
                    data = None
        except Exception:
            data = None

        try:
            context.storage_state(path=storage_state_path)
        except Exception:
            pass
        context.close()
        browser.close()

        if data:
            return data
        else:
            raise RuntimeError('Failed to fetch option chain from NSE (blocked or endpoint changed)')

# ----------------------------- MATH / MODEL -----------------------------
def bs_prob_finish_itm(S: float, K: float, r: float, T: float, sigma: float, option_type: str) -> float:
    """Return probability that option finishes ITM at expiry using Black-Scholes assumption."""
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

# ----------------------------- ANALYSIS -----------------------------
def analyze_option_chain(data: Dict, symbol: str) -> pd.DataFrame:
    records = []
    underlying = data.get('records', {}).get('underlyingValue') or data.get('records', {}).get('underlying')
    expiries = data.get('records', {}).get('expiryDates', [])
    options = data.get('records', {}).get('data', [])

    for row in options:
        strike = float(row.get('strikePrice'))
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

    # Parse expiry into timezone-aware UTC timestamps when possible (robust)
    df['expiry_dt'] = pd.to_datetime(df['expiry'], errors='coerce', utc=True)

    return df

def select_candidate_strikes(df: pd.DataFrame, capital: float, risk_pct: float) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    if df.empty:
        return candidates

    if 'expiry_dt' in df.columns:
        df['expiry_dt'] = pd.to_datetime(df['expiry_dt'], errors='coerce', utc=True)
        try:
            if df['expiry_dt'].dt.tz is None:
                df['expiry_dt'] = df['expiry_dt'].dt.tz_localize('UTC')
        except Exception:
            def _ensure_aware(x):
                if pd.isna(x):
                    return x
                if x.tzinfo is None:
                    return x.replace(tzinfo=pd.Timestamp.utcnow().tz)
                return x
            df['expiry_dt'] = df['expiry_dt'].apply(_ensure_aware)

    if SELECT_EXPIRY:
        sel_dt = pd.to_datetime(SELECT_EXPIRY, errors='coerce', utc=True)
        if pd.notna(sel_dt) and 'expiry_dt' in df.columns:
            df = df[df['expiry_dt'].dt.date == sel_dt.date()].copy()

    if (not SELECT_EXPIRY) and ('expiry_dt' in df.columns and df['expiry_dt'].notna().any()):
        nearest_expiry = df['expiry_dt'].min()
        df = df[df['expiry_dt'] == nearest_expiry].copy()

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
    df['atm_n'] = 1 - norm_col(df['atm_diff'])  # nearer ATM -> higher

    if 'expiry_dt' in df.columns and df['expiry_dt'].notna().any():
        today = pd.Timestamp.now(tz='UTC').normalize()
        try:
            if df['expiry_dt'].dt.tz is None:
                df['expiry_dt'] = df['expiry_dt'].dt.tz_localize('UTC')
            else:
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

    probs = []
    for _, r in df.iterrows():
        p = bs_prob_finish_itm(S, r['strike'], 0.06, float(r['T']), r['iv_f'], r['type'])
        probs.append(p)
    df['prob_itm'] = probs

    # -------- Directional OI logic: Call ΔOI vs Put ΔOI --------
    call_doi_total = float(df[df['type'] == 'CE']['oi_change'].sum())
    put_doi_total = float(df[df['type'] == 'PE']['oi_change'].sum())

    # Classify pattern based on your rules:
    # Buy CE when: Call OI ↓ and Put OI ↑
    # Buy PE when: Put OI ↓ and Call OI ↑
    # Avoid buying when: both OI ↑ or both OI ↓
    if call_doi_total < 0 and put_doi_total > 0:
        dir_bias = "BULL_FOR_CE"
    elif call_doi_total > 0 and put_doi_total < 0:
        dir_bias = "BEAR_FOR_PE"
    elif (call_doi_total > 0 and put_doi_total > 0) or (call_doi_total < 0 and put_doi_total < 0):
        dir_bias = "SIDEWAYS"
    else:
        dir_bias = "NEUTRAL"

    # Map this to a per-row directional score (0..1)
    df['dir_oi_score'] = 0.5  # neutral default
    if dir_bias == "BULL_FOR_CE":
        df.loc[df['type'] == 'CE', 'dir_oi_score'] = 1.0
        df.loc[df['type'] == 'PE', 'dir_oi_score'] = 0.0
    elif dir_bias == "BEAR_FOR_PE":
        df.loc[df['type'] == 'PE', 'dir_oi_score'] = 1.0
        df.loc[df['type'] == 'CE', 'dir_oi_score'] = 0.0
    elif dir_bias == "SIDEWAYS":
        df['dir_oi_score'] = 0.0  # avoid buying: no directional edge

    # -------- Score composition using feature toggles --------
    score_components = []
    if FACTOR_ENABLE.get('oi', False):
        score_components.append(FACTOR_WEIGHTS.get('oi', 0.0) * df['oi_n'])
    if FACTOR_ENABLE.get('oi_change', False):
        score_components.append(FACTOR_WEIGHTS.get('oi_change', 0.0) * df['oi_ch_n'])
    if FACTOR_ENABLE.get('volume', False):
        score_components.append(FACTOR_WEIGHTS.get('volume', 0.0) * df['vol_n'])
    if FACTOR_ENABLE.get('atm_proximity', False):
        score_components.append(FACTOR_WEIGHTS.get('atm_proximity', 0.0) * df['atm_n'])
    if FACTOR_ENABLE.get('prob_itm', False):
        score_components.append(FACTOR_WEIGHTS.get('prob_itm', 0.0) * df['prob_itm'])
    if FACTOR_ENABLE.get('directional_oi', False):
        score_components.append(FACTOR_WEIGHTS.get('directional_oi', 0.0) * df['dir_oi_score'])

    if score_components:
        df['score'] = sum(score_components)
    else:
        df['score'] = 0.0

    # Filter low OI or vol
    df = df[(df['oi'] >= MIN_OI) & (df['vol'] >= MIN_VOLUME)].copy()
    if df.empty:
        return candidates

    MAX_TOTAL = int(MAX_CANDIDATES) if MAX_CANDIDATES > 0 else 1

    if FORCE_ONE_OF_EACH:
        ce_df = df[df['type'] == 'CE'].sort_values(['score', 'prob_itm'], ascending=False)
        pe_df = df[df['type'] == 'PE'].sort_values(['score', 'prob_itm'], ascending=False)

        chosen_rows = []
        if not ce_df.empty:
            chosen_rows.append(ce_df.iloc[0])
        if not pe_df.empty:
            chosen_rows.append(pe_df.iloc[0])

        remaining_slots = MAX_TOTAL - len(chosen_rows)
        if remaining_slots > 0:
            drop_idx = [int(r.name) for r in chosen_rows]
            others = df.drop(index=drop_idx, errors='ignore').sort_values(['score', 'prob_itm'], ascending=False).head(remaining_slots)
            chosen_rows.extend([row for _, row in others.iterrows()])

        if chosen_rows:
            top_rows = pd.DataFrame(chosen_rows)
        else:
            top_rows = pd.DataFrame(columns=df.columns)
    else:
        top_rows = df.sort_values(['score', 'prob_itm'], ascending=False).head(MAX_TOTAL)

    # ---------------- Build candidate dicts ----------------
    for _, r in top_rows.iterrows():
        try:
            ltp_val = float(r['ltp'])
            if math.isnan(ltp_val) or ltp_val <= 0:
                premium = 0.5
            else:
                premium = max(0.5, ltp_val)
        except Exception:
            premium = 0.5

        stop_loss_price = premium * 0.5
        target_price = premium * (1 + TARGET_RR)
        risk_amount = (premium - stop_loss_price)
        allowed_risk = (capital * (risk_pct / 100.0))
        denom = (risk_amount if (risk_amount and risk_amount > 0) else premium)
        qty_units = int(max(1, math.floor(allowed_risk / denom))) if denom > 0 else 1

        # Determine lot size to convert units -> lots
        lot_size = get_lot_size_for_symbol(SYMBOL)

        if ROUND_TO_LOTS:
            lots = max(MIN_LOTS, qty_units // lot_size)
            if lots == 0 and qty_units >= lot_size:
                lots = 1
            qty = int(lots * lot_size)
        else:
            qty = qty_units
            lots = math.ceil(qty_units / lot_size) if lot_size > 0 else 0

        # Base conviction from score
        conviction = float(min(99.9, 100 * float(r.get('score', 0.0))))

        # If global OI pattern is sideways (both OI ↑ or both ↓), reduce conviction
        if dir_bias == "SIDEWAYS":
            conviction *= 0.5  # you can tweak this penalty

        # Compose reason: show active factor contributions and OI pattern
        contribs = {}
        if FACTOR_ENABLE.get('oi', False):
            contribs['OI'] = FACTOR_WEIGHTS.get('oi', 0.0) * float(r['oi_n'])
        if FACTOR_ENABLE.get('oi_change', False):
            contribs['OI_change'] = FACTOR_WEIGHTS.get('oi_change', 0.0) * float(r['oi_ch_n'])
        if FACTOR_ENABLE.get('volume', False):
            contribs['Volume'] = FACTOR_WEIGHTS.get('volume', 0.0) * float(r['vol_n'])
        if FACTOR_ENABLE.get('atm_proximity', False):
            contribs['ATM_proximity'] = FACTOR_WEIGHTS.get('atm_proximity', 0.0) * float(r['atm_n'])
        if FACTOR_ENABLE.get('prob_itm', False):
            contribs['Prob_ITM'] = FACTOR_WEIGHTS.get('prob_itm', 0.0) * float(r['prob_itm'])
        if FACTOR_ENABLE.get('directional_oi', False):
            contribs['Directional_OI'] = FACTOR_WEIGHTS.get('directional_oi', 0.0) * float(r['dir_oi_score'])

        sorted_contrib = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        top_factors = ', '.join([f"{name}({value:.3f})" for name, value in sorted_contrib[:3]]) if sorted_contrib else 'No factors enabled'

        # Short human-readable tag for the OI pattern
        if dir_bias == "BULL_FOR_CE":
            dir_comment = "Directional OI bias: Call OI ↓ & Put OI ↑ → favourable for CE buying."
        elif dir_bias == "BEAR_FOR_PE":
            dir_comment = "Directional OI bias: Put OI ↓ & Call OI ↑ → favourable for PE buying."
        elif dir_bias == "SIDEWAYS":
            dir_comment = "Directional OI bias: Both Call & Put OI moving in same direction → avoid aggressive option buying."
        else:
            dir_comment = "Directional OI bias: Neutral / mixed."

        reason_parts = [
            f"Chosen because top combined score among strikes for expiry {r.get('expiry')}.",
            f"Top factors: {top_factors}.",
            f"Underlying={S}, Strike={r['strike']}, LTP={premium}, IV={float(r['iv_f']*100):.2f}%, Prob_ITM={float(r['prob_itm']):.3%}.",
            f"OI={int(r['oi'])}, ΔOI={int(r['oi_change'])}, Volume={int(r['vol'])}, ATM_diff={abs(r['atm_diff']):.2f}.",
            f"Total ΔOI context: Call ΔOI={call_doi_total:+.0f}, Put ΔOI={put_doi_total:+.0f}. {dir_comment}",
            f"Score={float(r['score']):.4f}, Conviction={conviction:.1f}%.",
            f"Trade sizing => Qty={qty} units ({lots} lots of size {lot_size}) for risk {risk_pct}% of capital ({allowed_risk}). "
            f"Stop={round(stop_loss_price,2)}, Target={round(target_price,2)} (R:R={TARGET_RR}:1)."
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
            'stop_loss': round(stop_loss_price, 2),
            'target': round(target_price, 2),
            'rr': TARGET_RR,
            'expiry': str(r.get('expiry')),
            'T_days': int(r.get('T_days', -1)),
            'reason': reason_text
        })

    return candidates

# ----------------------------- GOOGLE SHEETS LOG -----------------------------
def log_to_google_sheet(creds_json: str, sheet_name: str, rows: List[Dict]):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
    client = gspread.authorize(creds)

    try:
        sheet = client.open(sheet_name).sheet1
    except Exception:
        sh = client.create(sheet_name)
        sheet = sh.sheet1

    header = ['timestamp', 'symbol', 'type', 'strike', 'ltp', 'qty', 'lots', 'lot_size', 'oi', 'oi_change', 'iv',
              'prob_itm', 'conviction_pct', 'stop_loss', 'target', 'rr', 'expiry', 'T_days', 'reason']
    try:
        existing = sheet.row_values(1)
        if not existing or existing[0].lower() != 'timestamp':
            sheet.insert_row(header, 1)
    except Exception:
        pass

    for r in rows:
        row = [datetime.utcnow().isoformat(), r['symbol'], r['type'], r['strike'], r['ltp'], r['qty'], r.get('lots'),
               r.get('lot_size'), r['oi'], r['oi_change'], r['iv'], r['prob_itm'], r['conviction_pct'],
               r['stop_loss'], r['target'], r['rr'], r['expiry'], r['T_days'], r['reason']]
        sheet.append_row(row)

# ----------------------------- MAIN -----------------------------
def main():
    now = datetime.now()
    if not is_market_day(now):
        print('Today is not a market day. Exiting.')
        return

    print(f'Fetching option chain for {SYMBOL}...')
    data = fetch_option_chain_playwright(SYMBOL)
    df = analyze_option_chain(data, SYMBOL)
    if df.empty:
        print('No valid option data parsed. Exiting.')
        return

    candidates = select_candidate_strikes(df, CAPITAL, RISK_PER_TRADE_PCT)
    if not candidates:
        print('No candidates found after filtering. Try relaxing MIN_OI or MIN_VOLUME.')
        return

    print('Top trade suggestions:')
    for c in candidates:
        print(json.dumps(c, default=str))

    # log to Google Sheets
    if GS_CREDS_JSON and os.path.exists(GS_CREDS_JSON):
        try:
            log_to_google_sheet(GS_CREDS_JSON, GOOGLE_SHEET_NAME, candidates)
            print('Logged suggestions to Google Sheet.')
        except Exception as e:
            print('Failed to log to Google Sheet:', e)
    else:
        print('Google Sheets credentials not found or path invalid. Skipping logging.')

if __name__ == '__main__':
    main()