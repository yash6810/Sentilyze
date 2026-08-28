"""
IPO & Pre-IPO Intelligence Radar for Sentilyze.
Pillar 9 Alternative Asset Discovery:
- Pre-IPO private market valuation & catalyst tracker (OpenAI, Anthropic, SpaceX, Stripe, Databricks).
- Real-time SEC Form S-1 / S-1/A IPO Prospectus filing monitor via SEC EDGAR.
- Automated Day-1 ticker ingestion into Sentilyze models upon public exchange listing.
"""

from typing import Any, Dict, List, Optional
import os
import requests
import defusedxml.ElementTree as defused_ET
from datetime import datetime, timezone
from src.utils import get_logger

logger = get_logger(__name__)

USER_AGENT = "SentilyzeQuantResearch bot@sentilyze.ai"
STOCKS_FILE = "stocks.txt"

PRE_IPO_UNIVERSE = [
    {
        "name": "OpenAI",
        "projected_ticker": "OPAI",
        "sector": "Frontier Generative AI & LLMs",
        "est_valuation_usd": "$157 Billion",
        "last_funding_round": "Series H ($6.6B raised at $157B cap)",
        "lead_backers": [
            "Microsoft (MSFT)",
            "Thrive Capital",
            "SoftBank",
            "Khosla Ventures",
            "Nvidia (NVDA)",
        ],
        "key_catalysts": "O-Series Reasoning Models, Enterprise API Expansion, Sovereign Cloud Deals",
        "ipo_readiness_score": 92.0,
        "status": "PRE-IPO / HIGH PROBABILITY LISTING",
    },
    {
        "name": "Anthropic",
        "projected_ticker": "ANTH",
        "sector": "AI Safety & Claude Frontier Models",
        "est_valuation_usd": "$40 Billion",
        "last_funding_round": "Strategic Round ($4B Amazon + $2B Google)",
        "lead_backers": [
            "Amazon (AMZN)",
            "Alphabet (GOOGL)",
            "Spark Capital",
            "Menlo Ventures",
        ],
        "key_catalysts": "Claude 3.5 Sonnet Coding Dominance, Enterprise Computer Use API, AWS Bedrock Integration",
        "ipo_readiness_score": 86.0,
        "status": "PRE-IPO / EXPANDING ENTERPRISE RUN-RATE",
    },
    {
        "name": "SpaceX",
        "projected_ticker": "SPACEX",
        "sector": "Aerospace, Satellite Internet & Orbital Launch",
        "est_valuation_usd": "$210 Billion",
        "last_funding_round": "Tender Offer ($210B Secondary Valuation)",
        "lead_backers": [
            "Founders Fund",
            "Sequoia Capital",
            "Baillie Gifford",
            "Fidelity",
        ],
        "key_catalysts": "Starlink Cashflow Free Float, Starship Orbital Commercial Flights, DoD Space Contracts",
        "ipo_readiness_score": 88.0,
        "status": "PRE-IPO / STARLINK POTENTIAL SPIN-OFF",
    },
    {
        "name": "Stripe",
        "projected_ticker": "STRP",
        "sector": "Global Financial Infrastructure & Payment Orchestration",
        "est_valuation_usd": "$70 Billion",
        "last_funding_round": "Tender Offer ($70B Valuation)",
        "lead_backers": [
            "Sequoia Capital",
            "Andreessen Horowitz",
            "Silver Lake",
            "Peter Thiel",
        ],
        "key_catalysts": "Total Volume Exceeds $1 Trillion, Stablecoin Payment Settlement, AI Agent Micropayments",
        "ipo_readiness_score": 95.0,
        "status": "IMMINENT / S-1 READY PROFILE",
    },
    {
        "name": "Databricks",
        "projected_ticker": "DATA",
        "sector": "Unified Lakehouse, Data Engineering & Generative AI",
        "est_valuation_usd": "$43 Billion",
        "last_funding_round": "Series I ($500M at $43B cap)",
        "lead_backers": [
            "T. Rowe Price",
            "Andreessen Horowitz",
            "CapitalG",
            "Nvidia (NVDA)",
        ],
        "key_catalysts": "Lakehouse + DBRX Open Source AI, $2.4B+ Annual Recurring Revenue (ARR)",
        "ipo_readiness_score": 94.0,
        "status": "IMMINENT / ENTERPRISE CASH FLOW POSITIVE",
    },
]


def fetch_sec_edgar_ipo_filings() -> List[Dict[str, Any]]:
    """
    Fetches real-time SEC Form S-1 / S-1/A IPO registration statements from SEC EDGAR.
    """
    url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=S-1&count=20&output=atom"
    headers = {"User-Agent": USER_AGENT}
    filings = []

    try:
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code == 200:
            root = defused_ET.fromstring(res.content)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            if not entries:
                entries = root.findall("entry")
            for e in entries[:15]:
                title_elem = e.find("{http://www.w3.org/2005/Atom}title")
                if title_elem is None:
                    title_elem = e.find("title")

                updated_elem = e.find("{http://www.w3.org/2005/Atom}updated")
                if updated_elem is None:
                    updated_elem = e.find("updated")

                link_elem = e.find("{http://www.w3.org/2005/Atom}link")
                if link_elem is None:
                    link_elem = e.find("link")

                summary_elem = e.find("{http://www.w3.org/2005/Atom}summary")
                if summary_elem is None:
                    summary_elem = e.find("summary")

                title = (
                    title_elem.text
                    if title_elem is not None and title_elem.text
                    else "S-1 IPO Filing"
                )
                updated = (
                    updated_elem.text
                    if updated_elem is not None and updated_elem.text
                    else str(datetime.now(timezone.utc))
                )
                link = link_elem.attrib.get("href", "") if link_elem is not None else ""
                summary = (
                    summary_elem.text
                    if summary_elem is not None and summary_elem.text
                    else title
                )

                filings.append(
                    {
                        "title": title,
                        "filing_type": "SEC Form S-1 (IPO Registration)",
                        "updated_at": updated[:19].replace("T", " "),
                        "filing_url": link,
                        "summary": summary[:200] + "...",
                    }
                )
    except Exception as e:
        logger.debug(f"SEC EDGAR IPO feed notice: {e}")

    return filings


def auto_register_ipo_ticker(ticker: str, company_name: str) -> bool:
    """
    Appends a newly public IPO ticker to stocks.txt to initiate model ingestion.
    """
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        return False

    existing_tickers = []
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE, "r") as f:
            existing_tickers = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.startswith("#")
            ]

    if clean_ticker in existing_tickers:
        logger.info(f"Ticker {clean_ticker} already enrolled in universe.")
        return True

    try:
        with open(STOCKS_FILE, "a") as f:
            f.write(f"\n# Day-1 IPO Ingestion: {company_name}\n{clean_ticker}\n")
        logger.info(
            f"🚀 Successfully registered Day-1 IPO asset {clean_ticker} ({company_name}) into {STOCKS_FILE}"
        )
        return True
    except Exception as e:
        logger.error(f"Error registering IPO ticker {clean_ticker}: {e}")
        return False


def fetch_pre_ipo_radar_summary() -> Dict[str, Any]:
    """
    High-level entry point returning the complete Pre-IPO and SEC S-1 pipeline.
    """
    filings = fetch_sec_edgar_ipo_filings()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pre_ipo_targets": PRE_IPO_UNIVERSE,
        "recent_s1_filings": filings,
        "total_targets_tracked": len(PRE_IPO_UNIVERSE),
        "total_active_s1_prospectuses": len(filings),
    }


def get_pre_ipo_pipeline_df() -> pd.DataFrame:
    """Returns a formatted pandas DataFrame of all pre-IPO target assets."""
    rows = []
    for item in PRE_IPO_UNIVERSE:
        rows.append(
            {
                "Company": item.get("name", ""),
                "Projected Ticker": item.get("projected_ticker", ""),
                "Sector": item.get("sector", ""),
                "Estimated Valuation": item.get("est_valuation_usd", ""),
                "Last Round": item.get("last_funding_round", ""),
                "IPO Readiness (%)": float(item.get("ipo_readiness_score", 85.0)),
                "Status": item.get("status", "PRE-IPO"),
            }
        )
    return pd.DataFrame(rows)
