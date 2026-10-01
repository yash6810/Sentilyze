"""
DuckDB & Apache Arrow In-Memory Zero-Copy Analytical Store (Sprint 4, Module 4.1 / Idea 44)
Accelerates backtesting, trade telemetry, and universe scanning by replacing slow CSV parsing
with zero-copy columnar Apache Arrow and DuckDB SQL queries.
"""

from typing import Dict, Any, List, Optional, Union
import os
import glob
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_STORE_DIR = os.path.join("data", "parquet_store")


class DuckDBStore:
    """
    In-Memory & Persistent Parquet + Apache Arrow Zero-Copy Store.
    Provides sub-millisecond SQL queries over historical bars, trades, and portfolio states.
    """

    def __init__(self, store_dir: str = DEFAULT_STORE_DIR):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        # Dedicated in-memory connection with Arrow integration
        self.con = duckdb.connect(database=":memory:")

    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Zero-copy register a pandas DataFrame into DuckDB as an Arrow-backed virtual table.
        """
        arrow_table = pa.Table.from_pandas(df)
        self.con.register(table_name, arrow_table)
        logger.debug(f"Registered table '{table_name}' with {len(df)} rows.")

    def save_dataframe_as_parquet(self, df: pd.DataFrame, file_basename: str) -> str:
        """
        Saves a DataFrame as an optimized Snappy-compressed columnar Parquet file.
        """
        if not file_basename.endswith(".parquet"):
            file_basename += ".parquet"
        target_path = os.path.join(self.store_dir, file_basename)
        arrow_table = pa.Table.from_pandas(df)
        pq.write_table(arrow_table, target_path, compression="SNAPPY")
        return target_path

    def query(self, sql: str) -> pd.DataFrame:
        """
        Executes arbitrary SQL and returns results as a pandas DataFrame via zero-copy Arrow bridge.
        """
        return self.con.execute(sql).df()

    def query_parquet_file(self, parquet_path: str, sql: str) -> pd.DataFrame:
        """
        Directly queries a Parquet file without loading into memory first:
        Example: SELECT * FROM read_parquet('data.parquet') WHERE Close > 100
        """
        safe_path = parquet_path.replace("\\", "/")
        formatted_sql = sql.replace("{FILE}", f"'{safe_path}'")
        return self.con.execute(formatted_sql).df()

    def compute_vectorized_equity_curve_metrics(
        self, returns_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculates institutional backtest metrics (CAGR, Sharpe, Max Drawdown)
        in pure DuckDB SQL window functions.
        """
        if returns_df.empty or "return" not in returns_df.columns:
            return {"status": "INVALID_DATA"}

        self.register_dataframe("strategy_returns", returns_df)

        sql = """
        WITH numbered AS (
            SELECT 
                row_number() OVER () AS t,
                return
            FROM strategy_returns
        ),
        cum_series AS (
            SELECT 
                t,
                return,
                exp(sum(ln(1.0 + return)) OVER (ORDER BY t)) AS cum_wealth
            FROM numbered
        ),
        peak_series AS (
            SELECT 
                t,
                return,
                cum_wealth,
                max(cum_wealth) OVER (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS peak_wealth
            FROM cum_series
        ),
        drawdown_series AS (
            SELECT 
                t,
                return,
                cum_wealth,
                peak_wealth,
                (cum_wealth - peak_wealth) / peak_wealth AS drawdown
            FROM peak_series
        )
        SELECT 
            count(*) AS total_bars,
            avg(return) AS mean_return,
            stddev_samp(return) AS std_return,
            min(drawdown) AS max_drawdown,
            last(cum_wealth) - 1.0 AS total_return
        FROM drawdown_series;
        """
        res = self.con.execute(sql).df()
        if res.empty:
            return {"status": "EMPTY_RESULT"}

        row = res.iloc[0]
        mean_ret = float(row["mean_return"]) if not pd.isna(row["mean_return"]) else 0.0
        std_ret = float(row["std_return"]) if not pd.isna(row["std_return"]) else 1e-4
        sharpe = (mean_ret / max(std_ret, 1e-6)) * (252.0**0.5)
        mdd = float(row["max_drawdown"]) if not pd.isna(row["max_drawdown"]) else 0.0
        tot_ret = (
            float(row["total_return"]) if not pd.isna(row["total_return"]) else 0.0
        )

        return {
            "status": "SUCCESS",
            "total_bars": int(row["total_bars"]),
            "annualized_sharpe": round(float(sharpe), 2),
            "max_drawdown_pct": round(float(mdd) * 100.0, 2),
            "total_return_pct": round(float(tot_ret) * 100.0, 2),
        }
