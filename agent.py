"""Omega Calmar — mean-reversion alpha wrapped in a Calmar-first risk shell.

  1. Market regime switch (SPY/QQQ trend + vol) — ON / SOFT / OFF.
     Fast to de-risk, slow to re-risk (asymmetric confirmation). The single
     biggest Calmar lever: don't catch falling knives in a bear tape.
  2. Self-drawdown governor — scales gross down in bands as our own equity
     pulls back from its peak, so a bad stretch can't compound into a deep hole.
  3. Market-vol targeting — gross scales inversely with realized SPY vol, so the
     book is small exactly when the denominator would otherwise blow out.
  4. ATR stop-loss exits — cut losers using avg_cost, alongside the existing
     z-score / RSI take-profit on winners.
  5. Asymmetric exposure — lever up toward the cap when calm and trending,
     flatten to cash when not. Maximizes the ratio, not just the numerator.

Long-only, beta-adjusted gross capped at 1.45x. Pure NumPy, no network, no deps.
"""
import numpy as np

def _sma(arr, n):
    if len(arr) < n: return np.mean(arr)
    return np.mean(arr[-n:])

def _std(arr, n):
    if len(arr) < n: return np.std(arr) if len(arr) > 1 else 1e-6
    return np.std(arr[-n:])

def _atr(high, low, close, n=14):
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    if len(tr) < n: return np.mean(tr)
    return np.mean(tr[-n:])

def _rsi(close, period=14):
    if len(close) < period + 1: return 50.0
    deltas = np.diff(close[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss < 1e-12: return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

def _hurst(close, max_lags=10):
    if len(close) < max_lags * 2: return 0.50
    try:
        lags = np.arange(2, max_lags)
        variances = []
        for lag in lags:
            diffs = close[lag:] - close[:-lag]
            s = np.std(diffs)
            variances.append(s if s > 0 else 1e-6)
        poly = np.polyfit(np.log(lags), np.log(variances), 1)
        return float(np.clip(poly[0] * 2.0, 0.0, 1.0))
    except Exception:
        return 0.50

def _ann_vol(close, n=20):
    if len(close) < n + 1: return None
    rets = np.diff(close[-(n + 1):]) / close[-(n + 1):-1]
    if len(rets) < 2: return None
    return float(np.std(rets) * np.sqrt(252))

def _ret(close, days):
    if len(close) < days + 1 or close[-(days + 1)] <= 0: return None
    return float(close[-1] / close[-(days + 1)] - 1.0)

BETA_3X = {"TQQQ", "SOXL", "UPRO", "SPXL", "TNA", "FAS", "TECL", "LABU", "CURE", "DRN", "UDOW", "NAIL"}
BETA_2X = {"QLD", "SSO", "DDM", "ROM", "UWM", "AGQ"}

def _beta(t):
    return 3.0 if t in BETA_3X else 2.0 if t in BETA_2X else 1.0

_regime = "soft"
_pending = None
_pending_n = 0
_peak_equity = 0.0

def _raw_regime(market_state):
    spy = market_state.get("SPY")
    qqq = market_state.get("QQQ")
    if not spy or len(spy) < 60:
        return "soft"
    sc = np.array([b["close"] for b in spy], dtype=float)
    qc = np.array([b["close"] for b in qqq], dtype=float) if qqq and len(qqq) >= 60 else sc

    r3 = _ret(sc, 3)
    v10 = _ann_vol(sc, 10)
    if (r3 is not None and r3 < -0.045) or (v10 is not None and v10 > 0.42):
        return "off"

    r126 = _ret(sc, 126)
    v20 = _ann_vol(sc, 20)
    if r126 is not None and v20 is not None and r126 < -0.10 and v20 > 0.30:
        return "off"

    sma50 = _sma(sc, 50)
    sma200 = _sma(sc, 200) if len(sc) >= 200 else None
    qsma50 = _sma(qc, 50)
    if v20 is None:
        return "soft"

    above_long = (sma200 is None) or (sc[-1] > sma200)
    if sc[-1] > sma50 * 1.003 and qc[-1] > qsma50 * 1.003 and v20 < 0.32 and above_long:
        return "on"
    return "soft"

def _confirm(raw):
    global _regime, _pending, _pending_n
    if raw == "off":                      
        _regime, _pending, _pending_n = "off", None, 0
        return "off"
    if raw == _regime:
        _pending, _pending_n = None, 0
        return _regime
    need = 1 if (_regime == "off" and raw == "soft") else 2  
    if raw == _pending:
        _pending_n += 1
    else:
        _pending, _pending_n = raw, 1
    if _pending_n >= need:
        _regime, _pending, _pending_n = _pending, None, 0
    return _regime

def _vol_target_scale(market_state):
    spy = market_state.get("SPY")
    if not spy or len(spy) < 21:
        return 1.0
    sc = np.array([b["close"] for b in spy], dtype=float)
    v = _ann_vol(sc, 20)
    if not v or v <= 0:
        return 1.0
    return float(np.clip(0.16 / v, 0.35, 1.0))   

def _dd_scale(equity):
    global _peak_equity
    _peak_equity = max(_peak_equity, equity)
    if _peak_equity <= 0:
        return 1.0
    dd = 1.0 - equity / _peak_equity
    if dd >= 0.10: return 0.25
    if dd >= 0.06: return 0.50
    if dd >= 0.035: return 0.75
    return 1.0

def _reversion_score(bars):
    n = len(bars)
    if n < 30: return None
    close = np.array([b["close"] for b in bars], dtype=float)
    high = np.array([b["high"] for b in bars], dtype=float)
    low = np.array([b["low"] for b in bars], dtype=float)
    price = close[-1]
    if price <= 0: return None

    mean_20 = _sma(close, 20)
    std_20 = _std(close, 20)
    if std_20 < 1e-6: return None
    z_score = (price - mean_20) / std_20
    rsi = _rsi(close)
    lower_bb = mean_20 - 2.0 * std_20
    bb_dist = (price - lower_bb) / std_20
    hurst = _hurst(close[-30:])
    atr = _atr(high, low, close, 14)
    atr_pct = atr / price

    if n >= 100:
        sma100 = _sma(close, 100)
        if price < sma100 * 0.88: return None
    if n >= 50:
        sma50 = _sma(close, 50)
        if price < sma50 * 0.80: return None    

    is_oversold = (z_score < -0.7 or rsi < 38 or bb_dist < 0.3)
    if not is_oversold: return None

    score = 0.0
    score += max(0, -z_score) * 15.0
    score += max(0, (40.0 - rsi)) * 1.0
    score += max(0, -bb_dist) * 8.0
    if hurst < 0.45: score += (0.45 - hurst) * 30.0
    if hurst > 0.55: score *= 0.5
    if n > 3 and close[-1] / close[-2] - 1 > 0: score += 3.0   
    return {"score": score, "atr_pct": atr_pct, "price": price,
            "z_score": z_score, "rsi": rsi, "hurst": hurst}

def _quality_score(bars):
    n = len(bars)
    if n < 50: return None
    close = np.array([b["close"] for b in bars], dtype=float)
    high = np.array([b["high"] for b in bars], dtype=float)
    low = np.array([b["low"] for b in bars], dtype=float)
    price = close[-1]
    if price <= 0: return None
    atr = _atr(high, low, close, 14)
    atr_pct = atr / price
    vol = atr_pct * np.sqrt(252)
    if vol > 0.30: return None
    ret_20 = close[-1] / close[-20] - 1 if n >= 20 else 0
    if ret_20 < 0: return None
    rsi = _rsi(close)
    if rsi > 70: return None
    score = (1.0 / (vol + 0.05)) + ret_20 * 10
    return {"score": score, "atr_pct": atr_pct, "price": price,
            "z_score": 0, "rsi": rsi, "hurst": 0.50}

MAX_POSITIONS = 12
MAX_PER_NAME = 0.18          
GROSS_ON = 1.45            
GROSS_SOFT = 0.60
GROSS_OFF = 0.0
TARGET_RISK = 0.045
STOP_ATR_MULT = 2.5         

def _positions_map(portfolio_state):
    positions = {}
    for t, v in portfolio_state.items():
        if t in ("cash", "last_prices", "positions"): continue
        qty = v.get("quantity", v.get("qty", 0)) if isinstance(v, dict) else v
        if qty and float(qty) > 0: positions[t] = {"qty": float(qty), "avg_cost": 0.0}
    for pos in portfolio_state.get("positions", []) or []:
        t = str(pos.get("ticker", "")).upper()
        q = float(pos.get("quantity", 0))
        if q > 0:
            positions[t] = {"qty": q, "avg_cost": float(pos.get("avg_cost", 0.0))}
    return positions

def decide(market_state, portfolio_state, cash):
    orders = []
    cash = float(cash)
    positions = _positions_map(portfolio_state)

    prices = {}
    for t, bars in market_state.items():
        if bars: prices[t] = bars[-1]["close"]

    equity = cash
    for t, p in positions.items():
        if t in prices: equity += p["qty"] * prices[t]
    if equity <= 0: return orders

    regime = _confirm(_raw_regime(market_state))
    dd_scale = _dd_scale(equity)
    vt_scale = _vol_target_scale(market_state)

    if regime == "off":
        gross_cap = GROSS_OFF
    elif regime == "soft":
        gross_cap = GROSS_SOFT * dd_scale * vt_scale
    else:
        gross_cap = GROSS_ON * dd_scale * vt_scale

    for ticker, p in list(positions.items()):
        bars = market_state.get(ticker, [])
        if len(bars) < 20:
            if regime == "off":
                orders.append({"ticker": ticker, "side": "sell", "quantity": int(p["qty"])})
            continue
        close = np.array([b["close"] for b in bars], dtype=float)
        high = np.array([b["high"] for b in bars], dtype=float)
        low = np.array([b["low"] for b in bars], dtype=float)
        price = close[-1]
        mean_20 = _sma(close, 20)
        std_20 = _std(close, 20)
        z = (price - mean_20) / std_20 if std_20 > 1e-6 else 0
        rsi = _rsi(close)
        atr = _atr(high, low, close, 14)

        take_profit = z > 0.8 or rsi > 65
        stop_loss = p["avg_cost"] > 0 and price < p["avg_cost"] - STOP_ATR_MULT * atr
        if regime == "off" or take_profit or stop_loss:
            orders.append({"ticker": ticker, "side": "sell", "quantity": int(p["qty"])})

    if gross_cap <= 0:
        return orders

    held = set(positions)
    candidates = []
    for ticker, bars in market_state.items():
        if ticker in held: continue
        res = _reversion_score(bars)
        if res is not None:
            res["ticker"] = ticker
            candidates.append(res)
    if len(candidates) < 3:
        for ticker, bars in market_state.items():
            if ticker in held: continue
            res = _quality_score(bars)
            if res is not None:
                res["ticker"] = ticker
                candidates.append(res)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    slots = MAX_POSITIONS - len(positions)
    selected = candidates[:max(slots, 0)]

    exposure = sum(p["qty"] * prices[t] * _beta(t) for t, p in positions.items() if t in prices)
    target_exposure = equity * gross_cap

    for c in selected:
        room = target_exposure - exposure
        if room <= 0: break
        t = c["ticker"]
        stock_vol = c["atr_pct"] * np.sqrt(252)
        if stock_vol < 0.05: stock_vol = 0.15
        allocation = (TARGET_RISK / stock_vol) * equity
        allocation = min(allocation, equity * MAX_PER_NAME, room / _beta(t), cash * 0.45)
        qty = int(allocation / c["price"])
        if qty > 0:
            orders.append({"ticker": t, "side": "buy", "quantity": qty})
            cash -= qty * c["price"]
            exposure += qty * c["price"] * _beta(t)

    return orders
