"""
Workspace: Academic Quantitative Alpha Scanner UI
================================================
Directly pulls newly published quantitative finance preprints from arXiv:
- Statistical Finance (q-fin.ST)
- Trading and Market Microstructure (q-fin.TR)
- Portfolio Management (q-fin.PM)
"""

import streamlit as st
from src.arxiv_alpha_scanner import fetch_recent_quant_papers


def render_arxiv_research_workspace():
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">
                🔬 arXiv Quantitative Alpha & Microstructure Scanner
            </h3>
            <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                Continuous feed of cutting-edge academic preprints in market microstructure, algorithmic execution, and statistical arbitrage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_q, col_btn = st.columns([3, 1])
    with col_q:
        query_topic = st.selectbox(
            "📚 Research Focus Area",
            [
                "Market Microstructure & Order Flow",
                "Machine Learning & Deep Hedging",
                "Sentiment Analysis & NLP Alpha",
                "High-Frequency Execution & Limit Order Books",
                "Volatility Forecasting & Regime Switching",
            ],
            index=0,
        )
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        search_btn = st.button(
            "🔎 Fetch Preprints", key="btn_fetch_arxiv", use_container_width=True
        )

    with st.spinner("Fetching preprints from arXiv Quantitative Finance repository..."):
        papers = fetch_recent_quant_papers(topic=query_topic, max_results=6)

    st.markdown(
        f"**Found {len(papers)} Recent Academic Preprints for `{query_topic}`:**"
    )

    for p in papers:
        with st.container():
            st.markdown(
                f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <span style="font-weight: 700; color: #38BDF8; font-size: 0.95rem; line-height: 1.3;">
                            📄 {p.get('title')}
                        </span>
                        <span style="font-size: 0.75rem; color: #94A3B8; font-family: monospace; white-space: nowrap; margin-left: 12px;">
                            {p.get('published_date', '')[:10]}
                        </span>
                    </div>
                    <div style="font-size: 0.78rem; color: #A7F3D0; margin: 4px 0 8px 0;">
                        <strong>Authors:</strong> {p.get('authors', 'Quant Researchers')}
                    </div>
                    <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.45; max-height: 120px; overflow-y: auto; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px;">
                        {p.get('abstract')}
                    </div>
                    <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.72rem; color: #94A3B8; background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 4px;">
                            Category: {p.get('category', 'q-fin')}
                        </span>
                        <a href="{p.get('pdf_url', p.get('arxiv_url', '#'))}" target="_blank" style="font-size: 0.8rem; color: #10B981; text-decoration: none; font-weight: 600;">
                            📥 Read PDF Paper &rarr;
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
