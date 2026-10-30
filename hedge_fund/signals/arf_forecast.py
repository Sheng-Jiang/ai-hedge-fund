"""ARF 4-Layer Predictive Forecasting Alpha Model (The Barometer).

Forms point-in-time forward investment views based on:
- Layer 1: Supply-Chain Demand Pull & Expected Beat Margin (ĝ_ARF vs consensus)
- Layer 2: Revision Velocity (14d/30d analyst momentum & EPS surprise probability)
- Layer 3: Multi-Stage DCF Forward Fair-Value Fan (P10 Bear, P50 Base, P90 Bull)
- Layer 4: CYQ Microstructure Regime Breakout Probability & Chip Concentration

Connects directly to the `forecasts` table in ARF DuckDB (data/arf.db).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

logger = logging.getLogger(__name__)

DEFAULT_ARF_DB_PATH = Path(os.getenv("ARF_DB_PATH", "data/arf.db"))


class ARFForecastModel(QuantModel):
    """Alpha model powered by the 4-layer predictive forecasting engine.

    Unlike the lagging nowcast (ARFModel thermometer), ARFForecastModel acts as
    a forward-looking barometer predicting earnings beats, revision momentum,
    12-month fair-value fan upside, and microstructure breakout probabilities.
    """

    investment_approach = "long_short"

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        forecasts_df: pd.DataFrame | None = None,
        target_horizon_days: int = 90,
    ) -> None:
        self._target_horizon_days = target_horizon_days

        if forecasts_df is not None:
            self._df = forecasts_df.copy()
        else:
            path = Path(db_path) if db_path is not None else DEFAULT_ARF_DB_PATH
            self._df = self._load_from_db(path)

        if not self._df.empty:
            self._df["as_of_date"] = pd.to_datetime(self._df["as_of_date"])

    @property
    def name(self) -> str:
        return "arf_forecast"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        """Form a point-in-time view on *ticker* as of *date* using the 4-layer forecast."""
        if self._df.empty:
            return self._abstain(ticker, date, "No ARF forecast data loaded")

        ticker_clean = ticker.strip().upper()
        as_of = pd.to_datetime(date)

        visible = self._df[self._df["as_of_date"] <= as_of]
        matches = visible[visible["ticker"] == ticker_clean]
        if matches.empty:
            return self._abstain(
                ticker,
                date,
                f"{ticker_clean} has no ARF forecast on or before {date}",
            )

        # Sort by as_of_date descending to get the most recent point-in-time forecast
        matches = matches.sort_values("as_of_date", ascending=False)
        row = matches.iloc[0]
        forecast_date = row["as_of_date"].strftime("%Y-%m-%d")

        pred_rev_growth = self._safe_float(row.get("predicted_rev_growth"), 0.15)
        cons_rev_growth = row.get("consensus_rev_growth")
        cons_growth_val = self._safe_float(cons_rev_growth, 0.15) if pd.notna(cons_rev_growth) else None
        expected_beat = row.get("expected_beat_pct")
        expected_beat_val = self._safe_float(expected_beat, 0.0) if pd.notna(expected_beat) else 0.0

        surprise_prob = self._safe_float(row.get("eps_surprise_prob"), 0.50)
        v14 = row.get("revision_velocity_14d")
        v14_val = self._safe_float(v14, 0.0) if pd.notna(v14) else 0.0

        p10 = row.get("fair_value_p10")
        p50 = row.get("fair_value_p50")
        p90 = row.get("fair_value_p90")
        exp_return_p50 = row.get("expected_return_p50")
        exp_ret_val = self._safe_float(exp_return_p50, 0.0) if pd.notna(exp_return_p50) else 0.0

        breakout_prob = self._safe_float(row.get("regime_breakout_prob"), 0.50)
        chip_conc = row.get("chip_concentration_ratio")

        # ── Alpha Conviction Computation ──────────────────────────────────────
        # 1. Fundamental return upside to P50 (30% upside maps to 1.0)
        signal_return = float(np.clip(exp_ret_val / 0.30, -1.0, 1.0))

        # 2. Earnings beat margin (15% beat vs consensus maps to 1.0)
        signal_beat = float(np.clip(expected_beat_val / 0.15, -1.0, 1.0))

        # 3. Revision momentum (10% 14d revision maps to 0.5)
        signal_rev = float(np.clip(v14_val / 0.10, -0.5, 0.5))

        # Weighted raw conviction
        raw_conviction = (0.45 * signal_return) + (0.35 * signal_beat) + (0.20 * signal_rev)

        # 4. Microstructure breakout timing multiplier
        if breakout_prob >= 0.70 and raw_conviction > 0:
            raw_conviction *= 1.15
        elif breakout_prob <= 0.35 and raw_conviction > 0:
            raw_conviction *= 0.70

        conviction = self._normalize_to_signal(raw_conviction)

        # Build institutional reasoning
        bias = "Bullish" if conviction > 0.15 else ("Bearish" if conviction < -0.15 else "Neutral")
        beat_str = f"Expected Rev Beat {expected_beat_val:+.1%}" if expected_beat is not None else f"Pred Growth {pred_rev_growth:.1%}"
        p50_str = f", P50 Target Upside {exp_ret_val:+.1%}" if p50 is not None else ""
        breakout_str = f", Breakout Prob {breakout_prob:.0%}"
        reasoning = (
            f"ARF Barometer {bias} ({conviction:+.2f}): {beat_str}, "
            f"EPS Surprise Prob {surprise_prob:.0%}{p50_str}{breakout_str} "
            f"(forecast as of {forecast_date})."
        )

        components = {
            "predicted_rev_growth": round(pred_rev_growth, 4),
            "expected_beat_pct": round(expected_beat_val, 4),
            "eps_surprise_prob": round(surprise_prob, 4),
            "revision_velocity_14d": round(v14_val, 4),
            "expected_return_p50": round(exp_ret_val, 4),
            "regime_breakout_prob": round(breakout_prob, 4),
        }
        if cons_growth_val is not None:
            components["consensus_rev_growth"] = round(cons_growth_val, 4)
        if p10 is not None and not math.isnan(p10):
            components["fair_value_p10"] = round(float(p10), 2)
        if p50 is not None and not math.isnan(p50):
            components["fair_value_p50"] = round(float(p50), 2)
        if p90 is not None and not math.isnan(p90):
            components["fair_value_p90"] = round(float(p90), 2)
        if chip_conc is not None and not math.isnan(chip_conc):
            components["chip_concentration_ratio"] = round(float(chip_conc), 4)

        return Signal(
            model_name=self.name,
            ticker=ticker_clean,
            date=date,
            value=round(conviction, 2),
            reasoning=reasoning,
            components=components,
            metadata={
                "forecast_date": forecast_date,
                "horizon_days": int(row.get("horizon_days") or self._target_horizon_days),
                "model_version": str(row.get("model_version") or "v1.0"),
            },
        )

    def _abstain(self, ticker: str, date: str, reason: str) -> Signal:
        return Signal(
            model_name=self.name,
            ticker=ticker.strip().upper(),
            date=date,
            value=0.0,
            reasoning=reason,
            metadata={"abstained": True},
        )

    @staticmethod
    def _load_from_db(path: Path) -> pd.DataFrame:
        """Load forecast history from DuckDB."""
        if not path.exists():
            logger.warning("ARF DuckDB file not found at %s", path)
            return pd.DataFrame()
        try:
            con = duckdb.connect(str(path), read_only=True)
            try:
                query = "SELECT * FROM forecasts"
                return con.execute(query).df()
            finally:
                con.close()
        except Exception as exc:
            logger.error("Failed to load forecasts from %s: %s", path, exc)
            return pd.DataFrame()
