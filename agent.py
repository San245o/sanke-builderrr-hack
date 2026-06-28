    """IGNITION — explosive-leader bot with dip-entry, vol budget, and a parachute.

Thesis (what the brief actually rewards, done as a quant would):
  Ride the genuinely explosive names, but never *chase* them. Buy leaders on a
  pullback (buy low), trim them into strength (sell high), size every name
  INVERSELY to its volatility so the 150%-vol rockets stay small, and cut hard to
  cash/defensive the instant the index trend breaks. Calmar = return / max-DD, so
  the parachute and the vol budget matter as much as the upside.

Five engines:
  1. Regime parachute (SPY/QQQ trend + shock + vol)  -> ON / SOFT / OFF.
     Asymmetric: 1 bad signal de-risks, 2 good ones re-risk. Saves the denominator.
  2. Explosive selection: trend-gated blended momentum / vol. High momentum,
     low vol wins -> "risk-adjusted explosive", not "most volatile".
  3. Buy-low overlay: among qualifying *uptrend* leaders, prefer the ones that
     have dipped from their recent high and are short-term oversold; SKIP the
     extended/overbought ones. This is buy-low-sell-high *within* momentum.
  4. Vol-budget sizing: inverse-vol target risk per name, hard per-name cap,
     gross scaled by regime x drawdown-governor x market-vol-target.
  5. Exits: take-profit trim on overbought winners, ATR stop on losers, full
     liquidation on OFF. A small leveraged sleeve (SOXL/TQQQ/QLD) only in clean ON.

Long-only, beta-adjusted gross < 1.5x. Pure stdlib. No network / LLM / files.
"""
from __future__ import annotations
from statistics import pstdev

# ---- universe buckets -------------------------------------------------------
# Explosive single-name leaders (AI / chips / high-beta thematic)
EXPLOSIVE = (
    "NVDA", "AMD", "MU", "MRVL", "AVGO", "PLTR", "AMAT", "LRCX", "TSM", "QCOM",
    "DELL", "SMCI", "ARM", "COIN", "TSLA", "NFLX", "CRWD", "PANW", "APP", "HOOD",
    "SOFI", "VRT", "NBIS", "MSTR", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
)
# Thematic / index ETFs that ride the same theme with less single-name risk
THEME_ETF = ("SMH", "SOXX", "XLK", "QQQ", "XLC", "XLY")
# Quality / value strength — the rotation winners, lower vol (Calmar ballast)
QUALITY = ("UNH", "LLY", "JPM", "V", "MA", "XLV", "XLF", "XLI", "IWM")
# Defensive book used when SOFT/OFF
DEFENSIVE = ("XLP", "XLV", "XLU", "XLRE", "GLD", "TLT", "XLF")
# Leveraged sleeve — only in confirmed clean ON, tiny budget
LEV_SLEEVE = ("SOXL", "TQQQ", "QLD")

BETA_3X = {"TQQQ", "SOXL", "UPRO", "SPXL", "TNA", "FAS", "TECL", "LABU", "CURE", "DRN", "UDOW", "NAIL"}
BETA_2X = {"QLD", "SSO", "DDM", "ROM", "UWM", "AGQ"}

def _beta(t):
    return 3.0 if t in BETA_3X else 2.0 if t in BETA_2X else 1.0

# ---- knobs ------------------------------------------------------------------
MAX_POSITIONS = 8
NAME_CAP = 0.20            # explosive single name
ETF_CAP = 0.26            # diversified ETF can hold more
LEV_CAP = 0.10            # leveraged sleeve hard cap
GROSS_ON = 1.25
GROSS_SOFT = 0.45
GROSS_OFF = 0.0
TARGET_RISK = 0.075       # per-name vol budget (annualized) for inverse-vol sizing
REBALANCE_EVERY = 3
DEAD_BAND = 0.02

# regime thresholds
SHOCK_R3 = -0.045
SHOCK_V10 = 0.46
RECOVER_V20_MAX = 0.34

# drawdown governor bands
DD1, DD2, DD3 = 0.025, 0.045, 0.07

_ANN = 252 ** 0.5
_tick = 0
_last_rebalance = -10**9
_peak_equity = 0.0
_regime = "soft"
_pending = None
_pending_n = 0


def _closes(bars):
    out = []
    for b in bars or []:
        try:
            c = float(b["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if c <= 0:
            return []
        out.append(c)
    return out


def _highs(bars):
    try:
        return [float(b["high"]) for b in bars or []]
    except (KeyError, TypeError, ValueError):
        return []


def _lows(bars):
    try:
        return [float(b["low"]) for b in bars or []]
    except (KeyError, TypeError, ValueError):
        return []


def _sma(c, n):
    return sum(c[-n:]) / n if len(c) >= n else None


def _ret(c, days, skip=0):
    if len(c) < days + skip + 1:
        return None
    start = c[-(days + skip + 1)]
    end = c[-(skip + 1)] if skip else c[-1]
    return end / start - 1.0 if start > 0 else None


def _vol(c, n):
    if len(c) < n + 1:
        return None
    r = [c[i] / c[i - 1] - 1.0 for i in range(len(c) - n, len(c)) if c[i - 1] > 0]
    return pstdev(r) * _ANN if len(r) > 1 else None


def _rsi(c, period=14):
    if len(c) < period + 1:
        return 50.0
    gains = losses = 0.0
    for i in range(len(c) - period, len(c)):
        d = c[i] - c[i - 1]
        if d > 0:
            gains += d
        else:
            losses -= d
    if losses < 1e-12:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(highs, lows, closes, n=14):
    if len(closes) < 2:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-n:]) / min(len(trs), n) if trs else 0.0


def _positions(portfolio_state):
    out = {}
    for pos in portfolio_state.get("positions", []) or []:
        t = str(pos.get("ticker", "")).upper()
        if not t:
            continue
        try:
            q = float(pos.get("quantity", 0.0))
            ac = float(pos.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if q > 0:
            out[t] = {"quantity": q, "avg_cost": ac}
    return out


def _equity(portfolio_state, cash, prices):
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    for t, p in _positions(portfolio_state).items():
        px = prices.get(t) or p["avg_cost"]
        if px > 0:
            total += p["quantity"] * px
    return max(total, 0.0)


# ---- regime -----------------------------------------------------------------
def _raw_regime(market_state):
    spy = _closes(market_state.get("SPY") or [])
    qqq = _closes(market_state.get("QQQ") or [])
    if len(spy) < 60:
        return "soft"
    if not qqq or len(qqq) < 60:
        qqq = spy
    # shock -> OFF
    r3 = _ret(spy, 3)
    v10 = _vol(spy, 10)
    if (r3 is not None and r3 < SHOCK_R3) or (v10 is not None and v10 > SHOCK_V10):
        return "off"
    r126 = _ret(spy, 126)
    v20 = _vol(spy, 20)
    if r126 is not None and v20 is not None and r126 < -0.10 and v20 > 0.30:
        return "off"
    spy50 = _sma(spy, 50)
    qqq50 = _sma(qqq, 50)
    spy200 = _sma(spy, 200)
    if spy50 is None or qqq50 is None or v20 is None:
        return "soft"
    above_long = (spy200 is None) or (spy[-1] > spy200)
    # clean ON: both indices above their 50d with a margin, calm, above 200d
    if spy[-1] > spy50 * 1.003 and qqq[-1] > qqq50 * 1.003 and v20 < RECOVER_V20_MAX and above_long:
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
    need = 1 if (_regime == "off" and raw == "soft") else 2  # fast down, slow up
    if raw == _pending:
        _pending_n += 1
    else:
        _pending, _pending_n = raw, 1
    if _pending_n >= need:
        _regime, _pending, _pending_n = _pending, None, 0
    return _regime


def _dd_scale(equity):
    global _peak_equity
    _peak_equity = max(_peak_equity, equity)
    if _peak_equity <= 0:
        return 1.0
    dd = 1.0 - equity / _peak_equity
    if dd >= DD3:
        return 0.20
    if dd >= DD2:
        return 0.45
    if dd >= DD1:
        return 0.70
    return 1.0


def _mkt_vol_scale(market_state):
    spy = _closes(market_state.get("SPY") or [])
    v = _vol(spy, 20)
    if not v or v <= 0:
        return 1.0
    return max(0.40, min(1.0, 0.15 / v))


# ---- selection: explosive + buy-low ----------------------------------------
def _score(market_state, tickers, want_dip=True):
    """Trend-gated momentum / vol, with a buy-low overlay and overbought skip."""
    scored = []
    for t in tickers:
        c = _closes(market_state.get(t) or [])
        if len(c) < 60:
            continue
        r63 = _ret(c, 63, skip=5)     # 3m momentum, skip last week (reversal hygiene)
        r20 = _ret(c, 20)
        sma50 = _sma(c, 50)
        v = _vol(c, 20)
        if r63 is None or r20 is None or sma50 is None or v is None or v <= 0:
            continue
        # trend gate: only hold names in an uptrend
        if c[-1] <= sma50:
            continue
        # blended momentum must be positive (explosive = going up)
        mom = 0.55 * r63 + 0.45 * r20
        if mom <= 0:
            continue
        rsi = _rsi(c, 14)
        # overbought skip: don't chase parabolic names (sell-high discipline)
        if rsi > 78:
            continue
        hi20 = max(c[-20:])
        from_hi = c[-1] / hi20 - 1.0   # <=0; how far below recent high
        # core risk-adjusted explosive score
        score = mom / max(v, 0.10)
        if want_dip:
            # buy-low overlay: reward a healthy pullback (best around -3%..-12%),
            # reward short-term oversold, penalize extended/at-the-high names.
            dip = -from_hi  # positive when below high
            if 0.02 <= dip <= 0.14:
                score *= 1.0 + min(dip * 4.0, 0.6)      # up to +60%
            elif dip > 0.20:
                score *= 0.6                            # too deep = maybe broken
            if rsi < 45:
                score *= 1.20                           # oversold leader = prime buy-low
            elif rsi > 68:
                score *= 0.80                           # extended = wait
        scored.append((score, t, v))
    scored.sort(reverse=True)
    return scored


def _build_targets(market_state, equity, regime, gross_cap):
    if gross_cap <= 0:
        return {}
    if regime == "on":
        pool = EXPLOSIVE + THEME_ETF + QUALITY
        scored = _score(market_state, pool, want_dip=True)
    else:  # soft -> lean defensive + quality, still vol-budgeted
        scored = _score(market_state, QUALITY + DEFENSIVE + THEME_ETF, want_dip=True)
    if not scored:
        return {}
    picks = scored[:MAX_POSITIONS]

    # inverse-vol weights from a per-name risk budget (capped), distribute gross
    raw = {}
    for score, t, v in picks:
        w = TARGET_RISK / max(v, 0.10)
        cap = LEV_CAP if t in LEV_SLEEVE else ETF_CAP if t in THEME_ETF or t in DEFENSIVE else NAME_CAP
        raw[t] = min(w, cap)
    tot = sum(raw.values())
    if tot <= 0:
        return {}
    targets = {}
    for t, w in raw.items():
        cap = LEV_CAP if t in LEV_SLEEVE else ETF_CAP if t in THEME_ETF or t in DEFENSIVE else NAME_CAP
        targets[t] = min(cap, gross_cap * w / tot)
    # add a small leveraged sleeve in clean ON if room and calm
    if regime == "on" and gross_cap >= GROSS_ON * 0.8:
        lev = _score(market_state, LEV_SLEEVE, want_dip=True)
        if lev:
            lt = lev[0][1]
            targets[lt] = targets.get(lt, 0.0) + min(LEV_CAP, 0.06)
    return targets


# ---- exits & order construction --------------------------------------------
def _exit_orders(market_state, positions, regime):
    orders = []
    for t, p in positions.items():
        c = _closes(market_state.get(t) or [])
        if len(c) < 20:
            if regime == "off":
                orders.append({"ticker": t, "side": "sell", "quantity": int(p["quantity"])})
            continue
        highs = _highs(market_state.get(t) or [])
        lows = _lows(market_state.get(t) or [])
        price = c[-1]
        rsi = _rsi(c, 14)
        sma20 = _sma(c, 20)
        atr = _atr(highs, lows, c, 14) if highs and lows else 0.0
        z = (price - sma20) / (pstdev(c[-20:]) or 1e-9) if len(c) >= 20 else 0.0
        # take-profit: sell-high on extended winners
        take_profit = rsi > 74 or z > 2.3
        # stop-loss: cut losers vs avg cost using ATR
        stop = p["avg_cost"] > 0 and atr > 0 and price < p["avg_cost"] - 2.5 * atr
        # trend-break: fell below 50d
        sma50 = _sma(c, 50)
        broke = sma50 is not None and price < sma50 * 0.97
        if regime == "off" or take_profit or stop or broke:
            orders.append({"ticker": t, "side": "sell", "quantity": int(p["quantity"])})
    return orders


def _rebalance_orders(targets, positions, prices, equity, cash):
    orders = []
    min_trade = DEAD_BAND * equity
    sell_proceeds = 0.0
    held = set(positions)
    # sell names not in target (already partly handled by exits) and trims
    for t, p in positions.items():
        px = prices.get(t)
        if not px:
            continue
        delta = equity * targets.get(t, 0.0) - p["quantity"] * px
        if t not in targets:
            continue  # exits handle full liquidation of dropped names
        if delta < -min_trade:
            sq = min(int(abs(delta) // px), int(p["quantity"]))
            if sq > 0:
                orders.append({"ticker": t, "side": "sell", "quantity": sq})
                sell_proceeds += sq * px
    spendable = max(float(cash or 0.0), 0.0) + sell_proceeds * 0.99
    for t, w in sorted(targets.items(), key=lambda kv: kv[1], reverse=True):
        px = prices.get(t)
        if not px:
            continue
        cur = positions.get(t, {}).get("quantity", 0.0)
        delta = equity * w - cur * px
        if delta < min_trade:
            continue
        bq = int(min(delta, spendable) // px)
        if bq > 0:
            orders.append({"ticker": t, "side": "buy", "quantity": bq})
            spendable -= bq * px
    return orders


def decide(market_state, portfolio_state, cash):
    global _tick, _last_rebalance
    _tick += 1
    cash = float(cash)

    prices = {}
    for t, bars in market_state.items():
        c = _closes(bars)
        if c:
            prices[str(t).upper()] = c[-1]

    positions = _positions(portfolio_state)
    equity = _equity(portfolio_state, cash, prices)
    if equity <= 0:
        return []

    regime = _confirm(_raw_regime(market_state))
    dd = _dd_scale(equity)
    mv = _mkt_vol_scale(market_state)
    if regime == "off":
        gross_cap = GROSS_OFF
    elif regime == "soft":
        gross_cap = GROSS_SOFT * dd * mv
    else:
        gross_cap = GROSS_ON * dd * mv

    # exits run every day (urgent); rebalance on cadence
    exits = _exit_orders(market_state, positions, regime)
    urgent = regime == "off" or bool(exits)
    if regime == "off":
        return exits[:45]

    if not urgent and _tick - _last_rebalance < REBALANCE_EVERY:
        return exits[:45]

    # remove exited names from the position map before rebalancing
    exit_names = {o["ticker"] for o in exits}
    remaining = {t: p for t, p in positions.items() if t not in exit_names}

    targets = _build_targets(market_state, equity, regime, gross_cap)
    rebal = _rebalance_orders(targets, remaining, prices, equity, cash)
    orders = exits + rebal
    if orders:
        _last_rebalance = _tick
    return orders[:45]
