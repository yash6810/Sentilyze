---
name: sentilyze-audit-protocol
description: Standing audit protocol for Sentilyze. Run before reporting any task as complete to check for fabricated/synthetic data, ticker-invariant bugs, mislabeled methodology, results-file/README consistency, duplicate-output smells, safety-critical logic, and silent failures.
---

# Sentilyze — Standing Audit Protocol

Run this entire protocol yourself, on the actual current repo state, before reporting
any task as complete. Do not describe what you intended to do — verify what the code
actually does, and report findings the same way: confirmed, not verified, or broken.

Do not trust your own prior claims, your own comments, or your own README text as
evidence. Verify against the code and the persisted result files directly, every time.

---

## 1. Fabricated / fake data check

For every module that claims to source external or "real" data:
- Open the full function body, not just the signature.
- Does it make a real network call (`requests.get`, an SDK client, a real API)? Or does
  it return hardcoded values, `np.random.*` output, or a list literal dressed up as
  live data?
- If it's a fallback (real source fails → synthetic substitute), does it flag that
  fact in its return value (e.g. `is_real_data: bool`)? Does every caller of that
  function actually check the flag, or does the fake data flow through silently with
  the same shape as real data?
- Search for these smells specifically: `"calibrated realistic"`, `"synthetic"`,
  `"placeholder"`, `"dummy"`, `np.random.uniform(`, `np.random.randn(`, hardcoded
  numeric literals returned from a function that's supposed to compute something.

Report every match with the file, function name, and one sentence on whether it's a
legitimate fallback (flagged, logged, or clearly labeled) or a silent fabrication.

## 2. Ticker/input-invariant bugs

For any function that's supposed to produce a per-ticker, per-date, or per-config
result: pick 2+ different real inputs, run the function (or trace the call site), and
confirm the outputs actually differ. If a function is called with missing/default
arguments that make it ignore its real inputs, that's a bug — find every call site
where meaningful arguments should be passed but aren't.

Specifically re-check for: any place a `ticker` argument is accepted but not used to
select data; any place two named configurations are supposed to be different but a
constant or shared default causes them to collapse to the same computation.

## 3. Mislabeled methodology

If a function or class name claims a specific technique (PPO, Kelly Criterion, Sharpe
ratio, cointegration, etc.), verify the implementation actually matches that
technique's defining mechanism — not just that it uses similar terminology. Example
check: if something claims PPO, does it implement the clipped surrogate objective? If
a parameter for that mechanism exists (e.g. `clip_eps`) but is never referenced in the
update step, that's a mislabeling bug, not just a naming quibble.

## 4. Results-file / README consistency

For every table or claim in `README.md` that cites numbers from a `results/*.json`
file:
- Open both. Compare the actual values, not just the presence of a table.
- Check whether the stated methodology (e.g. "500-day evaluation window", "N Monte
  Carlo trials") matches what's actually in the JSON for every row, not just one.
- If they don't match exactly, the README is stale or was hand-written instead of
  generated from the data. Flag this explicitly and do not trust the README's
  narrative conclusions until the underlying file is confirmed current.

## 5. Duplicate-output smell test

For any study that runs multiple configurations/variants (ablations, comparisons,
A/B-style tests) across multiple entities (tickers, assets, scenarios): programmatically
compare the numeric output of every pair of configurations that are supposed to be
different. If any two configs produce identical results across all entities, that is
not a coincidence — it means one config isn't actually varying what it claims to vary.
Find and fix the root cause before trusting any conclusion drawn from that study.

## 6. Safety-critical logic sanity check

For any circuit breaker, kill switch, position limit, or risk control:
- Does it read live/current state (actual account balance, actual today's P&L), or
  does it compare against a hardcoded constant?
- Is the metric it checks actually scoped the way its name claims (e.g. a "daily loss"
  check should reset daily and measure realized+unrealized loss for that day
  specifically, not lifetime unrealized P&L)?
- Confirm it's actually wired into the decision path (grep isn't enough — trace the
  call site and confirm the returned boolean actually gates order placement).

## 7. Silent failure check

Search for `except Exception:` (or similarly broad catches) in any file that's part of
the live trading/decision path. For each one found:
- Does it log at a visible level (`warning`/`error`/`critical`), or only `debug`, or
  nothing at all (`pass`)?
- Could this exception hide a real failure that should stop or alert on the trading
  loop, or is it a genuinely low-stakes cleanup path (e.g. best-effort file deletion)?
- Report each one with a severity judgment, don't just list them.

## 8. Scope creep check

Count files in `src/`. Compare to the last audit's count (see `AUDIT_LOG.md` if
present). Any new module must have been explicitly requested — if the count grew
without an explicit task authorizing a new module, flag it and ask before continuing.

## 9. Unaudited-module sweep

Maintain a list of every file in `src/` and whether it's been through steps 1–7 above
at least once. Each audit run, pick up where the last one left off rather than
re-checking the same files repeatedly. Never claim a module is "clean" unless you've
actually opened and read it this session.

## 10. Strict Portfolio Preservation & Realized Gain Integrity

- Verify that `results/paper_portfolio.json` and `results/executed_trades.csv` are NEVER overwritten with dummy/test data.
- The portfolio baseline is **$152,198.09 Cash (100% Realized Gains, $0 debt, 0 open positions, 29 trades at 89.66% win rate)**.
- Any background test or CI script must use an isolated temporary portfolio directory (`tmp_path`) and must NEVER modify production `results/paper_portfolio.json`.


---

## Reporting format (use every time)

Structure the report exactly like this, every audit:

```
## Confirmed working (verified by reading full code / running the check)
- ...

## Confirmed broken (verified, with file/function/line evidence)
- ...

## Not verified this round (inference only, or not re-checked)
- ...

## Not opened this round (no claims made)
- ...
```

Do not mix these categories. Do not report a task as "done" unless it's in the first
bucket with specific evidence (file path, function name, and what you actually checked
— not what you intended to check).
