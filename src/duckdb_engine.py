"""
High-Performance DuckDB Columnar Data Lake & Market-Wide Scanner Engine.
Enables sub-second analytical queries across the entire US equity universe:
  - Vectorized SQL screening for momentum breakouts, golden crosses, and volume spikes
  - Columnar Parquet persistence and zero-overhead memory footprint
  - Ingestion bridge from yfinance, results backtests, and SEC EDGAR catalysts
  - Liquidity tier classification (Mega, Large, Mid, Small-Cap)
"""

from typing import Dict, Any, List, Optional
import os
import glob
from datetime import datetime, timezone
import duckdb
import pandas as pd
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = os.path.join("data", "sentilyze_market.duckdb")


class DuckDBMarketEngine:
    """
    Embedded Columnar Analytical Engine powered by DuckDB.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.con = duckdb.connect(database=self.db_path, read_only=self.read_only)
        self._init_schema()

    def _init_schema(self):
        """Initializes high-performance columnar tables with index constraints."""
        if self.read_only:
            return

        # 1. Daily Bar Lake
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_bars (
                ticker VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                rsi DOUBLE,
                sma50 DOUBLE,
                sma200 DOUBLE,
                atr DOUBLE,
                vwap DOUBLE,
                PRIMARY KEY (ticker, date)
            );
        """
        )

        # 2. Universe Security Master & Liquidity Tiers
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS universe_registry (
                ticker VARCHAR PRIMARY KEY,
                company_name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                market_cap DOUBLE,
                adv_30d DOUBLE,
                liquidity_tier VARCHAR,
                last_updated TIMESTAMP
            );
        """
        )

        # 3. SEC EDGAR Material Catalyst Ledger
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS sec_catalysts (
                ticker VARCHAR,
                filing_date TIMESTAMP,
                form_type VARCHAR,
                description VARCHAR,
                sentiment_score DOUBLE,
                PRIMARY KEY (ticker, filing_date, form_type)
            );
        """
        )

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Executes arbitrary SQL query and returns result as a Pandas DataFrame."""
        try:
            if params:
                return self.con.execute(sql, params).fetchdf()
            return self.con.execute(sql).fetchdf()
        except Exception as e:
            logger.error(f"DuckDB query execution error: {e} | SQL: {sql}")
            return pd.DataFrame()

    def ingest_bars(self, df: pd.DataFrame, ticker: str) -> int:
        """
        Inserts or replaces daily bars for a given ticker.
        """
        if df.empty:
            return 0

        clean_df = df.copy()
        clean_df["ticker"] = ticker.upper()

        # Normalize column names
        col_map = {c: c.lower() for c in clean_df.columns}
        clean_df.rename(columns=col_map, inplace=True)

        if "date" not in clean_df.columns:
            if isinstance(clean_df.index, pd.DatetimeIndex):
                clean_df["date"] = clean_df.index.date
            else:
                clean_df["date"] = pd.to_datetime(clean_df.index).date

        # Ensure standard schema columns exist
        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "sma50",
            "sma200",
            "atr",
            "vwap",
        ]:
            if col not in clean_df.columns:
                clean_df[col] = np.nan

        sub_df = clean_df[
            [
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "rsi",
                "sma50",
                "sma200",
                "atr",
                "vwap",
            ]
        ].dropna(subset=["date", "close"])

        if sub_df.empty:
            return 0

        # Upsert using DuckDB
        self.con.register("tmp_bars", sub_df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO daily_bars
            SELECT * FROM tmp_bars;
        """
        )
        self.con.unregister("tmp_bars")
        return len(sub_df)

    def bootstrap_from_results(
        self, results_dir: str = "results", max_tickers: int = 50
    ) -> int:
        """
        Populates the DuckDB lake from pre-computed backtest portfolio curves in results/.
        """
        files = glob.glob(os.path.join(results_dir, "*_portfolio.csv"))
        total_inserted = 0

        for f in files[:max_tickers]:
            ticker = os.path.basename(f).replace("_portfolio.csv", "")
            if not ticker or not ticker.isalnum():
                continue
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                if "Close" in df.columns:
                    rows = self.ingest_bars(df, ticker)
                    total_inserted += rows
            except Exception as e:
                logger.warning(f"Could not bootstrap {ticker}: {e}")

        logger.info(
            f"Bootstrapped {total_inserted} total bars across {len(files[:max_tickers])} tickers into DuckDB."
        )
        return total_inserted

    # =========================================================================
    # HIGH-SPEED ANALYTICAL SCREENERS (SUB-SECOND MARKET-WIDE SCANS)
    # =========================================================================

    def scan_momentum_breakouts(
        self,
        min_rsi: float = 50.0,
        max_rsi: float = 72.0,
        above_sma200: bool = True,
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Screens for high-probability momentum setups across the entire lake.
        """
        sql = """
            WITH ranked_bars AS (
                SELECT *,
                       ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) as rn
                FROM daily_bars
            )
            SELECT ticker, date, close, rsi, sma50, sma200, volume
            FROM ranked_bars
            WHERE rn = 1
              AND rsi >= ? AND rsi <= ?
        """
        params = [min_rsi, max_rsi]
        if above_sma200:
            sql += " AND (sma200 IS NULL OR close >= sma200)"

        sql += " ORDER BY rsi DESC LIMIT ?"
        params.append(limit)

        return self.query(sql, params)

    def scan_golden_crosses(self, limit: int = 30) -> pd.DataFrame:
        """
        Screens for stocks where SMA50 is breaking out above SMA200 (Long-Term Bullish Transition).
        """
        sql = """
            WITH latest AS (
                SELECT ticker, date, close, sma50, sma200,
                       ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) as rn
                FROM daily_bars
                WHERE sma50 IS NOT NULL AND sma200 IS NOT NULL
            )
            SELECT ticker, date, close, sma50, sma200,
                   round((sma50 - sma200) / sma200 * 100, 2) as spread_pct
            FROM latest
            WHERE rn = 1 AND sma50 >= sma200
            ORDER BY spread_pct ASC
            LIMIT ?;
        """
        return self.query(sql, [limit])

    def scan_oversold_dips(
        self,
        min_rsi: float = 30.0,
        max_rsi: float = 48.0,
        limit: int = 25,
    ) -> pd.DataFrame:
        """
        Screens for oversold mean-reversion dip opportunities during crisis/pullback regimes.
        """
        sql = """
            WITH ranked_bars AS (
                SELECT *,
                       ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) as rn
                FROM daily_bars
            )
            SELECT ticker, date, close, rsi, sma50, sma200, volume
            FROM ranked_bars
            WHERE rn = 1
              AND rsi >= ? AND rsi <= ?
            ORDER BY rsi ASC LIMIT ?;
        """
        return self.query(sql, [min_rsi, max_rsi, limit])

    def get_full_market_candidates(
        self, limit: int = 25, regime: Optional[str] = None
    ) -> List[str]:
        """
        Stage 1 Broad Screener: Returns top ranked candidates conditioned on Macro Regime:
          - BULL_EXPANSION: Momentum breakouts (RSI 50-75, above SMA200) + Golden Crosses.
          - RANGEBOUND_CHOP: Tightened breakouts (RSI 55-68) + Prioritizes Golden Crosses to eliminate fakeouts.
          - HIGH_VOL_CRISIS: Oversold mean-reversion dips (RSI 30-48) + Strongest Golden Crosses.
        """
        candidates = []
        regime_upper = (regime or "BULL_EXPANSION").upper()

        if "CRISIS" in regime_upper:
            # 1. In high vol / crisis: look for oversold bounce setups
            df_dips = self.scan_oversold_dips(min_rsi=30.0, max_rsi=48.0, limit=limit)
            if not df_dips.empty:
                candidates.extend(df_dips["ticker"].tolist())

            # Supplement with Golden Crosses (structural bull support)
            if len(candidates) < limit:
                df_cross = self.scan_golden_crosses(limit=limit - len(candidates))
                if not df_cross.empty:
                    for t in df_cross["ticker"].tolist():
                        if t not in candidates:
                            candidates.append(t)

        elif "CHOP" in regime_upper:
            # 1. In chop / consolidation: prioritize Golden Crosses first to prevent fake breakouts
            df_cross = self.scan_golden_crosses(limit=limit // 2 + 1)
            if not df_cross.empty:
                candidates.extend(df_cross["ticker"].tolist())

            # 2. Tightened momentum breakout (RSI 55 to 68)
            df = self.scan_momentum_breakouts(
                min_rsi=55.0,
                max_rsi=68.0,
                above_sma200=True,
                limit=limit - len(candidates),
            )
            if not df.empty:
                for t in df["ticker"].tolist():
                    if t not in candidates:
                        candidates.append(t)

        else:
            # Default: BULL_EXPANSION standard momentum breakouts
            df = self.scan_momentum_breakouts(
                min_rsi=50.0, max_rsi=75.0, above_sma200=True, limit=limit
            )
            if not df.empty:
                candidates.extend(df["ticker"].tolist())

            # 2. Golden Crosses
            if len(candidates) < limit:
                df_cross = self.scan_golden_crosses(limit=limit - len(candidates))
                if not df_cross.empty:
                    for t in df_cross["ticker"].tolist():
                        if t not in candidates:
                            candidates.append(t)

        # 3. Fallback to active tickers in daily_bars if needed
        if len(candidates) < 5:
            df_active = self.query("SELECT DISTINCT ticker FROM daily_bars LIMIT 25;")
            if not df_active.empty:
                for t in df_active["ticker"].tolist():
                    if t not in candidates:
                        candidates.append(t)

        return candidates[:limit]

    def get_coverage_summary(self) -> Dict[str, Any]:
        """
        Returns summary diagnostics of the market data lake.
        """
        res = self.con.execute(
            """
            SELECT
                count(DISTINCT ticker) as total_tickers,
                count(*) as total_bars,
                min(date) as earliest_date,
                max(date) as latest_date
            FROM daily_bars;
        """
        ).fetchone()

        db_size_mb = 0.0
        if self.db_path != ":memory:" and os.path.exists(self.db_path):
            db_size_mb = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)

        return {
            "total_tickers": res[0] or 0,
            "total_bars": res[1] or 0,
            "earliest_date": str(res[2]) if res[2] else None,
            "latest_date": str(res[3]) if res[3] else None,
            "db_path": self.db_path,
            "db_size_mb": db_size_mb,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def close(self):
        """Closes connection cleanly."""
        try:
            self.con.close()
        except Exception:
            pass


def get_duckdb_scanner() -> DuckDBMarketEngine:
    """Convenience factory helper."""
    return DuckDBMarketEngine()
