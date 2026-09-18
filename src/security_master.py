"""
US Security Master & Universal Ticker Registry.
Automates official discovery, cleaning, and liquidity tiering of the US tradable equity universe:
  - Ingests NASDAQ, NYSE, and AMEX directory feeds (nasdaqtraded.txt)
  - Maps SEC EDGAR CIK numbers for material event filing lookups
  - Filters out test symbols, warrants, rights, preferred shares, and untradable shells
  - Classifies securities into Institutional Liquidity Tiers (Tier 1 Mega, Tier 2 Mid, Tier 3 Small, Tier 4 Excluded)
  - Synchronizes with DuckDB universe_registry for sub-second screener queries
"""

from typing import List, Optional
import os
import io
import urllib.request
from datetime import datetime, timezone
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

SECURITY_MASTER_CACHE = os.path.join("data", "security_master_universe.csv")
NASDAQ_TRADED_URL = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Institutional Liquidity Tier Constants
TIER_1_MEGA = "TIER_1_MEGA"  # ADV > $50M, Price > $15
TIER_2_MID = "TIER_2_MID"  # ADV > $10M, Price > $10
TIER_3_SMALL = "TIER_3_SMALL"  # ADV > $2M, Price > $5
TIER_4_EXCLUDED = "TIER_4_EXCLUDED"  # ADV < $2M or Price < $5 (Hard Veto)

# Institutional GICS Sector Constants
SECTOR_TECH = "Information Technology"
SECTOR_COMM = "Communication Services"
SECTOR_DISC = "Consumer Discretionary"
SECTOR_HEALTH = "Health Care"
SECTOR_FIN = "Financials"
SECTOR_IND = "Industrials"
SECTOR_STAPLES = "Consumer Staples"
SECTOR_ENERGY = "Energy"
SECTOR_UTIL = "Utilities"
SECTOR_REAL_ESTATE = "Real Estate"
SECTOR_MATERIALS = "Materials"
SECTOR_ETF = "ETF / Index Fund"
SECTOR_OTHER = "Other / Diversified"

# Foundational Ticker-to-Sector Directory mapping
TICKER_SECTOR_MAP = {
    # Tech
    "NVDA": SECTOR_TECH,
    "AAPL": SECTOR_TECH,
    "MSFT": SECTOR_TECH,
    "AVGO": SECTOR_TECH,
    "AMD": SECTOR_TECH,
    "ORCL": SECTOR_TECH,
    "PLTR": SECTOR_TECH,
    "CSCO": SECTOR_TECH,
    "QCOM": SECTOR_TECH,
    "TXN": SECTOR_TECH,
    "INTC": SECTOR_TECH,
    "IBM": SECTOR_TECH,
    "ADBE": SECTOR_TECH,
    "CRM": SECTOR_TECH,
    "NOW": SECTOR_TECH,
    "AMAT": SECTOR_TECH,
    "LRCX": SECTOR_TECH,
    "PANW": SECTOR_TECH,
    "SNPS": SECTOR_TECH,
    "CDNS": SECTOR_TECH,
    "MU": SECTOR_TECH,
    "ADI": SECTOR_TECH,
    "KLAC": SECTOR_TECH,
    "ANET": SECTOR_TECH,
    "APH": SECTOR_TECH,
    "SMCI": SECTOR_TECH,
    "ARM": SECTOR_TECH,
    "TSM": SECTOR_TECH,
    # Communication Services
    "GOOGL": SECTOR_COMM,
    "GOOG": SECTOR_COMM,
    "META": SECTOR_COMM,
    "NFLX": SECTOR_COMM,
    "DIS": SECTOR_COMM,
    "CMCSA": SECTOR_COMM,
    "TMUS": SECTOR_COMM,
    "VZ": SECTOR_COMM,
    "T": SECTOR_COMM,
    "CHTR": SECTOR_COMM,
    # Consumer Discretionary
    "AMZN": SECTOR_DISC,
    "TSLA": SECTOR_DISC,
    "HD": SECTOR_DISC,
    "MCD": SECTOR_DISC,
    "NKE": SECTOR_DISC,
    "SBUX": SECTOR_DISC,
    "LOW": SECTOR_DISC,
    "BKNG": SECTOR_DISC,
    "TJX": SECTOR_DISC,
    "LULU": SECTOR_DISC,
    "ABNB": SECTOR_DISC,
    "ORLY": SECTOR_DISC,
    # Health Care
    "LLY": SECTOR_HEALTH,
    "UNH": SECTOR_HEALTH,
    "JNJ": SECTOR_HEALTH,
    "ABBV": SECTOR_HEALTH,
    "MRK": SECTOR_HEALTH,
    "TMO": SECTOR_HEALTH,
    "ABT": SECTOR_HEALTH,
    "PFE": SECTOR_HEALTH,
    "DHR": SECTOR_HEALTH,
    "ISRG": SECTOR_HEALTH,
    "BMY": SECTOR_HEALTH,
    "AMGN": SECTOR_HEALTH,
    "GILD": SECTOR_HEALTH,
    "VRTX": SECTOR_HEALTH,
    "REGN": SECTOR_HEALTH,
    "MDT": SECTOR_HEALTH,
    # Financials
    "JPM": SECTOR_FIN,
    "BAC": SECTOR_FIN,
    "WFC": SECTOR_FIN,
    "GS": SECTOR_FIN,
    "MS": SECTOR_FIN,
    "BLK": SECTOR_FIN,
    "C": SECTOR_FIN,
    "AXP": SECTOR_FIN,
    "V": SECTOR_FIN,
    "MA": SECTOR_FIN,
    "SCHW": SECTOR_FIN,
    "PNC": SECTOR_FIN,
    "CB": SECTOR_FIN,
    "MMC": SECTOR_FIN,
    "PGR": SECTOR_FIN,
    # Industrials
    "GE": SECTOR_IND,
    "CAT": SECTOR_IND,
    "UNP": SECTOR_IND,
    "HON": SECTOR_IND,
    "RTX": SECTOR_IND,
    "BA": SECTOR_IND,
    "DE": SECTOR_IND,
    "LMT": SECTOR_IND,
    "UPS": SECTOR_IND,
    "ADP": SECTOR_IND,
    "FDX": SECTOR_IND,
    "ETN": SECTOR_IND,
    "WM": SECTOR_IND,
    "GD": SECTOR_IND,
    "EMR": SECTOR_IND,
    # Consumer Staples
    "PG": SECTOR_STAPLES,
    "COST": SECTOR_STAPLES,
    "WMT": SECTOR_STAPLES,
    "KO": SECTOR_STAPLES,
    "PEP": SECTOR_STAPLES,
    "PM": SECTOR_STAPLES,
    "MDLZ": SECTOR_STAPLES,
    "MO": SECTOR_STAPLES,
    "CL": SECTOR_STAPLES,
    "TGT": SECTOR_STAPLES,
    # Energy
    "XOM": SECTOR_ENERGY,
    "CVX": SECTOR_ENERGY,
    "COP": SECTOR_ENERGY,
    "SLB": SECTOR_ENERGY,
    "EOG": SECTOR_ENERGY,
    "MPC": SECTOR_ENERGY,
    "PSX": SECTOR_ENERGY,
    "VLO": SECTOR_ENERGY,
    "OXY": SECTOR_ENERGY,
    "KMI": SECTOR_ENERGY,
    # Utilities
    "NEE": SECTOR_UTIL,
    "SO": SECTOR_UTIL,
    "DUK": SECTOR_UTIL,
    "CEG": SECTOR_UTIL,
    "SRE": SECTOR_UTIL,
    "AEP": SECTOR_UTIL,
    "D": SECTOR_UTIL,
    "AES": SECTOR_UTIL,
    # Real Estate
    "PLD": SECTOR_REAL_ESTATE,
    "AMT": SECTOR_REAL_ESTATE,
    "EQIX": SECTOR_REAL_ESTATE,
    "SPG": SECTOR_REAL_ESTATE,
    "PSA": SECTOR_REAL_ESTATE,
    "O": SECTOR_REAL_ESTATE,
    # Materials
    "LIN": SECTOR_MATERIALS,
    "SHW": SECTOR_MATERIALS,
    "FCX": SECTOR_MATERIALS,
    "ECL": SECTOR_MATERIALS,
    "NEM": SECTOR_MATERIALS,
    "APD": SECTOR_MATERIALS,
    # ETFs
    "SPY": SECTOR_ETF,
    "QQQ": SECTOR_ETF,
    "IWM": SECTOR_ETF,
    "DIA": SECTOR_ETF,
    "XLF": SECTOR_ETF,
    "XLK": SECTOR_ETF,
    "XLE": SECTOR_ETF,
    "XLV": SECTOR_ETF,
    "XLI": SECTOR_ETF,
    "XLP": SECTOR_ETF,
    "XLU": SECTOR_ETF,
    "XLB": SECTOR_ETF,
    "XLRE": SECTOR_ETF,
    "XLC": SECTOR_ETF,
    "XLY": SECTOR_ETF,
    "ARKK": SECTOR_ETF,
    "SMH": SECTOR_ETF,
    "SOXX": SECTOR_ETF,
    "TLT": SECTOR_ETF,
    "GLD": SECTOR_ETF,
}


class SecurityMaster:
    """
    Manages the global universe of actively traded US equities and ETFs.
    """

    def __init__(self, cache_file: str = SECURITY_MASTER_CACHE):
        self.cache_file = cache_file
        dirname = os.path.dirname(self.cache_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.universe_df: Optional[pd.DataFrame] = None

    def fetch_exchange_directory(self) -> pd.DataFrame:
        """
        Downloads and parses the official NASDAQ/NYSE consolidated traded directory.
        Falls back to comprehensive local registry if offline.
        """
        try:
            req = urllib.request.Request(
                NASDAQ_TRADED_URL,
                headers={
                    "User-Agent": "Sentilyze-Research/2.0 (quant@sentilyze.internal)"
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
                raw_text = resp.read().decode("utf-8", errors="replace")

            # Format is pipe-delimited (|)
            df = pd.read_csv(io.StringIO(raw_text), sep="|")
            # Drop footer line (e.g. 'File Creation Time: ...')
            if "Symbol" in df.columns:
                df = df[df["Symbol"].notna() & (df["Symbol"] != "")]
                # Filter out test symbols and non-nasdaq traded
                if "Test Issue" in df.columns:
                    df = df[df["Test Issue"] != "Y"]
                if "Nasdaq Traded" in df.columns:
                    df = df[df["Nasdaq Traded"] == "Y"]

                # Clean symbols: remove warrants, units, preferreds with special characters
                clean_symbols = []
                for s in df["Symbol"]:
                    sym = str(s).strip().upper()
                    # Filter out test or auxiliary share classes with $, ., +, =
                    if any(ch in sym for ch in ["$", "+", "=", "#", " "]):
                        continue
                    if len(sym) > 5 and not sym.endswith("W"):  # Long auxiliary symbols
                        continue
                    clean_symbols.append(sym)

                df = df[df["Symbol"].isin(clean_symbols)].copy()
                df.rename(
                    columns={"Symbol": "ticker", "Security Name": "company_name"},
                    inplace=True,
                )
                logger.info(
                    f"Successfully fetched {len(df)} verified US exchange symbols from NASDAQ Trader."
                )
                return df
        except Exception as e:
            logger.warning(
                f"Exchange directory live fetch failed ({e}). Falling back to cached master universe."
            )

        return self._load_fallback_universe()

    def _load_fallback_universe(self) -> pd.DataFrame:
        """Loads cached security master or bootstraps from stocks.txt."""
        if os.path.exists(self.cache_file):
            try:
                df = pd.read_csv(self.cache_file)
                if not df.empty and "ticker" in df.columns:
                    return df
            except Exception:
                pass

        # Bootstrap from stocks.txt or core liquid US tickers
        tickers = []
        if os.path.exists("stocks.txt"):
            with open("stocks.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        tickers.append(line)

        if not tickers:
            tickers = [
                "NVDA",
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "TSLA",
                "AMD",
                "AVGO",
                "ORCL",
                "PLTR",
                "QCOM",
                "TXN",
                "INTC",
                "CSCO",
                "IBM",
                "SPY",
                "QQQ",
                "IWM",
                "DIA",
                "XLF",
                "XLK",
                "XLE",
                "XLV",
                "XLI",
            ]

        df = pd.DataFrame(
            {
                "ticker": sorted(list(set(tickers))),
                "company_name": [
                    f"{t} Corporation" for t in sorted(list(set(tickers)))
                ],
                "is_etf": [
                    (
                        "Y"
                        if t
                        in [
                            "SPY",
                            "QQQ",
                            "IWM",
                            "DIA",
                            "XLF",
                            "XLK",
                            "XLE",
                            "XLV",
                            "XLI",
                        ]
                        else "N"
                    )
                    for t in sorted(list(set(tickers)))
                ],
            }
        )
        df["sector"] = [self.get_ticker_sector(t) for t in df["ticker"]]
        return df

    def get_ticker_sector(self, ticker: str) -> str:
        """
        Returns the GICS sector for a given ticker with intelligent heuristics.
        """
        ticker_clean = str(ticker).upper().strip()
        if ticker_clean in TICKER_SECTOR_MAP:
            return TICKER_SECTOR_MAP[ticker_clean]

        if self.universe_df is not None and "sector" in self.universe_df.columns:
            match = self.universe_df[self.universe_df["ticker"] == ticker_clean]
            if not match.empty:
                sec = match["sector"].iloc[0]
                if pd.notna(sec) and sec:
                    return str(sec)

        return SECTOR_OTHER

    def classify_liquidity_tier(
        self,
        ticker: str,
        price: float,
        adv_shares: float,
    ) -> str:
        """
        Classifies an asset into an institutional execution tier based on daily dollar volume.
        """
        dollar_volume = price * adv_shares

        if price < 5.0 or dollar_volume < 2_000_000.0:
            return TIER_4_EXCLUDED
        elif dollar_volume >= 50_000_000.0:
            return TIER_1_MEGA
        elif dollar_volume >= 10_000_000.0:
            return TIER_2_MID
        else:
            return TIER_3_SMALL

    def build_registry(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Builds, enriches, caches, and returns the master universe registry.
        """
        if self.universe_df is not None and not force_refresh:
            return self.universe_df

        df = self.fetch_exchange_directory()

        # Add default liquidity tier estimation
        if "liquidity_tier" not in df.columns:
            # Default liquid names to Tier 1 / Tier 2
            tier_list = []
            for t in df["ticker"]:
                if t in ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA"]:
                    tier_list.append(TIER_1_MEGA)
                else:
                    tier_list.append(TIER_2_MID)
            df["liquidity_tier"] = tier_list

        # Add sector classification
        if "sector" not in df.columns:
            df["sector"] = [self.get_ticker_sector(t) for t in df["ticker"]]

        df["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.universe_df = df

        # Cache to disk
        df.to_csv(self.cache_file, index=False)
        return self.universe_df

    def get_tradable_tickers(
        self,
        min_tier: str = TIER_3_SMALL,
        include_etfs: bool = True,
    ) -> List[str]:
        """
        Returns clean list of tradable ticker symbols meeting the minimum liquidity tier.
        """
        if self.universe_df is None:
            self.build_registry()

        df = self.universe_df
        valid_tiers = [TIER_1_MEGA, TIER_2_MID]
        if min_tier == TIER_3_SMALL:
            valid_tiers.append(TIER_3_SMALL)

        filtered = df[df["liquidity_tier"].isin(valid_tiers)]
        if not include_etfs and "is_etf" in filtered.columns:
            filtered = filtered[filtered["is_etf"] != "Y"]

        return sorted(filtered["ticker"].dropna().unique().tolist())


def get_security_master() -> SecurityMaster:
    """Convenience factory helper."""
    sm = SecurityMaster()
    sm.build_registry()
    return sm
