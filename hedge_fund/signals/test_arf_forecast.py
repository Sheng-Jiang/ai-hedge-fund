"""Tests for ARFForecastModel (4-Layer Predictive Forecasting Alpha Model)."""

from __future__ import annotations

import pandas as pd
import pytest

from hedge_fund.signals import (
    ALPHA_MODEL_REGISTRY,
    ARFForecastModel,
    QuantModel,
    get_investment_approach,
)
from hedge_fund.signals.base import AlphaModel


class MockDataClient:
    """Mock DataClient for testing without external services."""
    pass


@pytest.fixture
def sample_forecasts_df() -> pd.DataFrame:
    """Fixture containing sample ARF forecasts across multiple dates."""
    return pd.DataFrame([
        # NVDA on 2026-06-01: High expected beat, high surprise prob, positive fair value upside
        {
            "ticker": "NVDA",
            "as_of_date": "2026-06-01",
            "horizon_days": 90,
            "predicted_rev_growth": 0.55,
            "consensus_rev_growth": 0.35,
            "expected_beat_pct": 0.20,
            "eps_surprise_prob": 0.88,
            "revision_velocity_14d": 0.08,
            "revision_velocity_30d": 0.15,
            "fair_value_p10": 110.0,
            "fair_value_p50": 150.0,
            "fair_value_p90": 200.0,
            "expected_return_p50": 0.25,
            "regime_breakout_prob": 0.82,
            "chip_concentration_ratio": 0.07,
            "model_version": "v1.0",
        },
        # NVDA on 2026-07-01 (later point in time)
        {
            "ticker": "NVDA",
            "as_of_date": "2026-07-01",
            "horizon_days": 90,
            "predicted_rev_growth": 0.58,
            "consensus_rev_growth": 0.38,
            "expected_beat_pct": 0.20,
            "eps_surprise_prob": 0.89,
            "revision_velocity_14d": 0.06,
            "revision_velocity_30d": 0.12,
            "fair_value_p10": 120.0,
            "fair_value_p50": 165.0,
            "fair_value_p90": 220.0,
            "expected_return_p50": 0.22,
            "regime_breakout_prob": 0.85,
            "chip_concentration_ratio": 0.06,
            "model_version": "v1.0",
        },
        # 300308.SZ on 2026-06-01: Optical supplier with demand pull beat
        {
            "ticker": "300308.SZ",
            "as_of_date": "2026-06-01",
            "horizon_days": 90,
            "predicted_rev_growth": 0.42,
            "consensus_rev_growth": 0.28,
            "expected_beat_pct": 0.14,
            "eps_surprise_prob": 0.78,
            "revision_velocity_14d": 0.05,
            "revision_velocity_30d": 0.09,
            "fair_value_p10": 150.0,
            "fair_value_p50": 210.0,
            "fair_value_p90": 280.0,
            "expected_return_p50": 0.18,
            "regime_breakout_prob": 0.72,
            "chip_concentration_ratio": 0.08,
            "model_version": "v1.0",
        },
        # INTC on 2026-06-01: Negative expected beat, negative revision velocity, low return
        {
            "ticker": "INTC",
            "as_of_date": "2026-06-01",
            "horizon_days": 90,
            "predicted_rev_growth": -0.05,
            "consensus_rev_growth": 0.05,
            "expected_beat_pct": -0.10,
            "eps_surprise_prob": 0.22,
            "revision_velocity_14d": -0.08,
            "revision_velocity_30d": -0.12,
            "fair_value_p10": 15.0,
            "fair_value_p50": 20.0,
            "fair_value_p90": 26.0,
            "expected_return_p50": -0.20,
            "regime_breakout_prob": 0.25,
            "chip_concentration_ratio": 0.22,
            "model_version": "v1.0",
        },
    ])


class TestARFForecastInterface:
    def test_registered_and_subclass(self):
        assert issubclass(ARFForecastModel, QuantModel)
        assert issubclass(ARFForecastModel, AlphaModel)
        assert "arf_forecast" in ALPHA_MODEL_REGISTRY
        assert ALPHA_MODEL_REGISTRY["arf_forecast"] is ARFForecastModel

    def test_name(self, sample_forecasts_df):
        model = ARFForecastModel(forecasts_df=sample_forecasts_df)
        assert model.name == "arf_forecast"

    def test_investment_approach(self):
        assert get_investment_approach("arf_forecast") == "long_short"


class TestARFForecastPredict:
    def test_bullish_forecast_generates_positive_conviction(self, sample_forecasts_df):
        model = ARFForecastModel(forecasts_df=sample_forecasts_df)
        sig = model.predict("NVDA", "2026-06-15", MockDataClient())
        assert sig.value > 0.50
        assert sig.model_name == "arf_forecast"
        assert "ARF Barometer Bullish" in sig.reasoning
        assert sig.components["expected_beat_pct"] == 0.20
        assert sig.components["eps_surprise_prob"] == 0.88
        assert sig.components["regime_breakout_prob"] == 0.82
        assert sig.components["fair_value_p50"] == 150.0

    def test_bearish_forecast_generates_negative_conviction(self, sample_forecasts_df):
        model = ARFForecastModel(forecasts_df=sample_forecasts_df)
        sig = model.predict("INTC", "2026-06-15", MockDataClient())
        assert sig.value < -0.30
        assert "ARF Barometer Bearish" in sig.reasoning
        assert sig.components["expected_beat_pct"] == -0.10
        assert sig.components["expected_return_p50"] == -0.20

    def test_point_in_time_filtering(self, sample_forecasts_df):
        model = ARFForecastModel(forecasts_df=sample_forecasts_df)
        # On 2026-06-15, should see 2026-06-01 forecast (fair_value_p50 = 150.0)
        sig_june = model.predict("NVDA", "2026-06-15", MockDataClient())
        assert sig_june.components["fair_value_p50"] == 150.0
        assert sig_june.metadata["forecast_date"] == "2026-06-01"

        # On 2026-07-15, should see 2026-07-01 forecast (fair_value_p50 = 165.0)
        sig_july = model.predict("NVDA", "2026-07-15", MockDataClient())
        assert sig_july.components["fair_value_p50"] == 165.0
        assert sig_july.metadata["forecast_date"] == "2026-07-01"

    def test_abstain_for_unknown_ticker(self, sample_forecasts_df):
        model = ARFForecastModel(forecasts_df=sample_forecasts_df)
        sig = model.predict("UNKNOWN", "2026-06-15", MockDataClient())
        assert sig.value == 0.0
        assert sig.metadata.get("abstained") is True

    def test_abstain_when_date_before_any_forecast(self, sample_forecasts_df):
        model = ARFForecastModel(forecasts_df=sample_forecasts_df)
        sig = model.predict("NVDA", "2026-01-01", MockDataClient())
        assert sig.value == 0.0
        assert sig.metadata.get("abstained") is True
