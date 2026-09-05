---
name: agent-browser
description: Automates real web browser interaction, live financial page navigation, form filling, and DOM extraction for market research, earnings calls, and regulatory filings.
---

# Agent Browser — Live Web Automation Skill

Use this skill when an agent needs to interact with interactive web pages, extract content from dynamic Single Page Applications (SPAs), download reports, or scrape financial portals when standard REST APIs are unavailable.

## 1. Core Principles
- **Headless Execution**: Use lightweight headless browsers (Playwright, Selenium, or CLI-based browser bridges) to navigate pages without spawning disruptive local UI windows.
- **Robust Selectors**: Prefer resilient CSS and XPath selectors (`data-testid`, semantic ARIA roles, or normalized text contents) over brittle auto-generated class names.
- **Rate-Limiting & Politeness**: Adhere to site `robots.txt`, include reasonable wait intervals, and set descriptive `User-Agent` headers.
- **Fail-Safe Fallbacks**: If interactive navigation times out or hits a captcha, gracefully degrade to raw static HTML parsing or cached fallback datasets.

## 2. Use Cases in Sentilyze
- **SEC EDGAR Filings**: Scraping 10-K / 10-Q filing sections directly from the SEC portal.
- **Federal Reserve & FOMC Transcripts**: Live extraction of Fed press conferences and rate statements.
- **Company Investor Relations**: Auto-navigating investor relations portals for recent press releases and earnings slides.
