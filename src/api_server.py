"""
Sentilyze Institutional Quant Engine - FastAPI Decoupled Backend Service.
Provides low-latency REST endpoints for Portfolio Ledgers, Global Market Sessions,
Multi-Agent Council Deliberations, Model Predictions, and Smartwatch Complications.
"""

import os
import csv
import json
import tracemalloc
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.utils import get_logger
from src.market_session import (
    get_regional_market_session,
    get_all_global_market_statuses,
    is_asset_market_open,
)
from src.currency_fx import get_usd_fx_rate, convert_to_usd
from src.smartwatch_api import generate_smartwatch_glance_payload

logger = get_logger(__name__)

# Start lightweight memory tracking
if not tracemalloc.is_tracing():
    tracemalloc.start()

app = FastAPI(
    title="Sentilyze Quant OS Engine API",
    version="3.0.0",
    description="High-Speed Decoupled Backend API for Multi-Market Quantitative Trading & Risk Management",
)

# CORS configuration for Streamlit, web clients, or mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORTFOLIO_FILE = os.path.join("results", "paper_portfolio.json")
TRADES_FILE = os.path.join("results", "executed_trades.csv")
START_TIME = datetime.now(timezone.utc)


@app.get("/")
def root_info() -> Dict[str, Any]:
    """Root info endpoint with service metadata and links."""
    return {
        "service": "Sentilyze Quant OS Engine API",
        "version": "3.0.0",
        "architecture": "Decoupled FastAPI Engine + Streamlit Viewport",
        "docs_url": "/docs",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "supported_theatres": ["US", "CRYPTO", "JAPAN", "INDIA", "CHINA_HK"],
    }


@app.get("/health")
@app.get("/api/v1/health")
def get_health() -> Dict[str, Any]:
    """System health, uptime, and resident memory footprint."""
    uptime_seconds = int((datetime.now(timezone.utc) - START_TIME).total_seconds())
    current_mem_mb = 0.0
    if tracemalloc.is_tracing():
        current_bytes, _ = tracemalloc.get_traced_memory()
        current_mem_mb = round(current_bytes / (1024 * 1024), 2)

    return {
        "status": "HEALTHY",
        "uptime_seconds": uptime_seconds,
        "memory_used_mb": current_mem_mb,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "global_sessions": get_all_global_market_statuses(),
    }


@app.get("/api/v1/portfolio")
def get_portfolio() -> Dict[str, Any]:
    """
    Returns real-time portfolio balance and positions from results/paper_portfolio.json.
    Strictly reads the source-of-truth ledger without mutating state.
    """
    if not os.path.exists(PORTFOLIO_FILE):
        raise HTTPException(status_code=404, detail="Portfolio state file not found.")

    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": "SUCCESS",
            "total_equity": data.get("total_equity", 100000.0),
            "cash": data.get("cash", 100000.0),
            "unrealized_pnl": data.get("unrealized_pnl", 0.0),
            "realized_pnl": data.get("realized_pnl", 0.0),
            "win_rate": data.get("win_rate", 0.0),
            "total_trades": data.get("total_trades", 0),
            "open_positions": data.get("open_positions", {}),
            "last_updated": data.get("last_updated"),
        }
    except Exception as e:
        logger.error(f"Error reading portfolio: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to read portfolio: {str(e)}"
        )


@app.get("/api/v1/trades")
def get_trades(
    limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """Returns paginated historical executed trades from results/executed_trades.csv."""
    if not os.path.exists(TRADES_FILE):
        return {"status": "SUCCESS", "total_records": 0, "trades": []}

    trades: List[Dict[str, Any]] = []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)

        total_records = len(trades)
        paginated = trades[offset : offset + limit]
        return {
            "status": "SUCCESS",
            "total_records": total_records,
            "limit": limit,
            "offset": offset,
            "trades": paginated,
        }
    except Exception as e:
        logger.error(f"Error reading trades: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read trades: {str(e)}")


@app.get("/api/v1/sessions")
def get_all_sessions() -> Dict[str, Any]:
    """Returns active session status for US, Crypto, Japan, India, and China/HK."""
    return {
        "status": "SUCCESS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "markets": get_all_global_market_statuses(),
    }


@app.get("/api/v1/session/{ticker}")
def get_ticker_session(ticker: str) -> Dict[str, Any]:
    """Returns the market session details for a specific ticker symbol."""
    sess = get_regional_market_session(ticker)
    return {
        "status": "SUCCESS",
        "ticker": ticker.upper(),
        "session": sess,
        "can_trade_now": is_asset_market_open(ticker),
    }


@app.get("/api/v1/fx/{currency}")
def get_fx_conversion(currency: str, amount: float = 1.0) -> Dict[str, Any]:
    """Converts a local currency amount into USD."""
    rate = get_usd_fx_rate(currency)
    converted_usd = convert_to_usd(amount, currency)
    return {
        "status": "SUCCESS",
        "currency": currency.upper(),
        "amount_local": amount,
        "rate_usd_per_unit": rate,
        "amount_usd": round(converted_usd, 4),
    }


@app.get("/api/v1/smartwatch")
def get_smartwatch_complication() -> Dict[str, Any]:
    """Returns structured complication payload for Apple Watch and Wear OS glance displays."""
    total_equity = 136545.29
    daily_pnl = 1.85
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                p_data = json.load(f)
                total_equity = float(p_data.get("total_equity", total_equity))
                daily_pnl = float(p_data.get("daily_return_pct", daily_pnl))
        except Exception:
            pass

    return generate_smartwatch_glance_payload(
        total_equity=total_equity,
        daily_pnl_pct=daily_pnl,
        top_active_ticker="NVDA",
        top_active_pnl_pct=2.1,
        top_signal_ticker="BTC-USD",
        top_signal_confidence=0.82,
    )


@app.post("/api/v1/execute_cycle")
def execute_autonomous_cycle(
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    force: bool = False,
) -> Dict[str, Any]:
    """
    Executes a single autonomous trading decision & risk evaluation cycle.
    Secured by optional X-API-KEY header.
    """
    expected_key = os.getenv("SENTILYZE_API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key.")

    try:
        from src.autonomous_trader import AutonomousTradingEngine

        engine = AutonomousTradingEngine()
        res = engine.run_autonomous_cycle()
        return {
            "status": "SUCCESS",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cycle_summary": res,
        }
    except Exception as e:
        logger.error(f"Error during cycle execution: {e}")
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Launches the Uvicorn server."""
    logger.info(f"🚀 Starting Sentilyze FastAPI Quant Engine on http://{host}:{port}")
    uvicorn.run(
        "src.api_server:app", host=host, port=port, reload=False, log_level="info"
    )


if __name__ == "__main__":
    start_server()
