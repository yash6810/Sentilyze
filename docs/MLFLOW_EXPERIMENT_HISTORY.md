# Sentilyze MLflow Experiment History & Historical Archive

> **Archival Date:** October 2, 2026  
> **Source Directory:** `mlruns/0`  
> **Total Archival Runs:** 7,587  
> **Total Unique Tickers Tested:** 6,261  
> **Original Artifact Footprint:** ~6.64 GB (6,801 MB)  
> **Full Machine-Readable CSV:** [`mlflow_runs_archive.csv`](./mlflow_runs_archive.csv)  

---

## 1. Executive Summary & Training Milestones

This archive documents the complete model training, hyperparameter optimization, and backtesting lineage across all experimental phases of the Sentilyze platform. All runs combine technical indicators (RSI, MACD, Bollinger Bands, Stochastic, VWAP) with FinBERT financial headline sentiment scores into native XGBoost classifiers.

### Training Phases Distribution

| Date | Runs Logged | Phase Description |
| :--- | :--- | :--- |
| **2026-03-06** | 8 | Initial baseline validation |
| **2026-03-12** | 14 | Initial baseline validation |
| **2026-03-13** | 7 | Initial baseline validation |
| **2026-03-14** | 8 | Initial baseline validation |
| **2026-08-23** | 53 | Feature pipeline & FinBERT v2 integration |
| **2026-08-26** | 2 | Feature pipeline & FinBERT v2 integration |
| **2026-08-28** | 100 | Feature pipeline & FinBERT v2 integration |
| **2026-08-29** | 2 | Feature pipeline & FinBERT v2 integration |
| **2026-09-05** | 564 | Multi-ticker momentum calibration |
| **2026-09-06** | 1 | Initial baseline validation |
| **2026-09-10** | 536 | Regime classification & dynamic leverage experiments |
| **2026-09-13** | 1 | Initial baseline validation |
| **2026-09-20** | 528 | Pre-deployment Walk-Forward Optimization (WFO) |
| **2026-09-21** | 61 | Initial baseline validation |
| **2026-09-22** | 5,681 | Massive 5,600+ ticker universal market universe sweep |
| **2026-09-26** | 12 | Production Walk-Forward retrainings & active council models |
| **2026-10-01** | 9 | Production Walk-Forward retrainings & active council models |

---

## 2. Core Benchmark Tickers Historical Performance

Summary of all recorded MLflow runs for the 7 primary production benchmark tickers:

| Ticker | Total Runs | Best Accuracy | Best ROC-AUC | Best Strategy Return | Best Buy & Hold Return |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVDA** | 29 | 52.80% | 0.5175 | 42020.09% | 3317.48% |
| **AAPL** | 19 | 50.59% | 0.4841 | 34514.17% | 514.25% |
| **MSFT** | 18 | 52.14% | 0.5128 | 403240.08% | 369.56% |
| **GOOGL** | 16 | 51.94% | 0.5227 | 152393.89% | 463.40% |
| **META** | 16 | 51.56% | 0.5188 | 572043.41% | 271.20% |
| **AMZN** | 14 | 51.30% | 0.5055 | 52545.00% | 174.19% |
| **TSLA** | 15 | 51.24% | 0.5047 | 120058.50% | 1721.34% |

### Detailed Breakdown by Benchmark Ticker

#### NVDA (29 Recorded Runs)
| Date | Run ID | Run Name | Accuracy | ROC-AUC | Return | Max Depth | LR | N Est |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-03-12 | `1acedf20` | vaunted-loon-664 | 49.75% | - | -0.15% | 4 | 0.05 | 200 |
| 2026-08-26 | `350707d9` | placid-snake-900 | 50.40% | 0.4907 | 423.39% | 3 | 0.03 | 150 |
| 2026-03-12 | `378a3a7b` | silent-shark-516 | 49.75% | - | 39342.74% | 4 | 0.05 | 200 |
| 2026-03-14 | `4240a784` | delicate-quail-441 | 50.20% | - | 37841.90% | 4 | 0.05 | 200 |
| 2026-03-14 | `4290d669` | delightful-dove-470 | 50.12% | - | 42020.09% | 4 | 0.05 | 200 |
| 2026-09-10 | `443be74d` | big-squirrel-619 | 52.73% | 0.5060 | 1844.28% | 3 | 0.03 | 150 |
| 2026-09-10 | `489e5ecd` | traveling-flea-5 | 52.80% | 0.5175 | 1898.81% | 3 | 0.03 | 150 |
| 2026-09-13 | `4a8e866a` | honorable-crab-257 | 52.73% | 0.5060 | 1844.28% | 3 | 0.03 | 150 |
| 2026-09-20 | `4ca0c277` | gentle-dove-350 | 52.21% | 0.5007 | 1844.28% | 3 | 0.03 | 150 |
| 2026-09-26 | `53e08340` | silent-cow-477 | 50.29% | 0.5131 | 2681.92% | 3 | 0.03 | 150 |

#### AAPL (19 Recorded Runs)
| Date | Run ID | Run Name | Accuracy | ROC-AUC | Return | Max Depth | LR | N Est |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-09-05 | `104538b4` | clean-fawn-301 | 50.59% | 0.4652 | 44.51% | 3 | 0.03 | 150 |
| 2026-03-13 | `120496e5` | defiant-loon-784 | 48.61% | - | 169.17% | 4 | 0.05 | 200 |
| 2026-03-12 | `1556bf0c` | rogue-goat-322 | 47.47% | - | 28191.26% | 4 | 0.05 | 200 |
| 2026-03-14 | `21341a5d` | peaceful-dolphin-514 | 48.61% | - | 34514.17% | 4 | 0.05 | 200 |
| 2026-08-23 | `24b67682` | loud-bat-919 | 48.41% | 0.4757 | 3.87% | 3 | 0.03 | 150 |
| 2026-09-20 | `368b44f9` | powerful-seal-549 | 50.13% | 0.4682 | 40.59% | 3 | 0.03 | 150 |
| 2026-08-23 | `3e073b78` | popular-cod-596 | 48.41% | 0.4757 | 3.87% | 3 | 0.03 | 150 |
| 2026-03-06 | `3f8998ce` | wise-hog-648 | 49.95% | - | - | 4 | 0.05 | 200 |
| 2026-08-23 | `4d9f31f6` | fearless-cod-828 | 48.61% | 0.4776 | 162.10% | 3 | 0.03 | 150 |
| 2026-09-26 | `50c66ff4` | adaptable-finch-909 | 49.90% | 0.4632 | 14.00% | 3 | 0.03 | 150 |

#### MSFT (18 Recorded Runs)
| Date | Run ID | Run Name | Accuracy | ROC-AUC | Return | Max Depth | LR | N Est |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-03-13 | `07950d6d` | trusting-kite-671 | 49.35% | - | 179.17% | 4 | 0.05 | 200 |
| 2026-08-23 | `1c6e08ee` | stately-doe-93 | 51.94% | 0.5111 | 150.59% | 3 | 0.03 | 150 |
| 2026-03-12 | `22453673` | gaudy-mink-676 | 49.85% | - | 33011.54% | 4 | 0.05 | 200 |
| 2026-08-23 | `23958112` | able-lamb-590 | 52.14% | 0.5128 | 168.63% | 3 | 0.03 | 150 |
| 2026-03-06 | `2567dedf` | inquisitive-grub-703 | 49.16% | - | -19.20% | 4 | 0.05 | 200 |
| 2026-08-23 | `4a2c71af` | traveling-gnu-604 | 51.44% | 0.5102 | 326.50% | 3 | 0.03 | 150 |
| 2026-03-14 | `4d1fa0be` | funny-grouse-109 | 49.35% | - | 403240.08% | 4 | 0.05 | 200 |
| 2026-08-23 | `582ecf30` | clumsy-lark-884 | 49.85% | 0.4968 | 172.96% | 4 | 0.05 | 200 |
| 2026-03-12 | `61b7d8d1` | funny-snail-937 | 49.85% | - | 33011.54% | 4 | 0.05 | 200 |
| 2026-08-23 | `6a1ea06e` | dashing-trout-166 | 51.94% | 0.5111 | 150.59% | 3 | 0.03 | 150 |

#### GOOGL (16 Recorded Runs)
| Date | Run ID | Run Name | Accuracy | ROC-AUC | Return | Max Depth | LR | N Est |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-08-23 | `0d87889d` | classy-gnat-294 | 51.94% | 0.5227 | 234.56% | 3 | 0.03 | 150 |
| 2026-08-23 | `14a0e1a7` | sassy-seal-694 | 51.94% | 0.5224 | 109.52% | 3 | 0.03 | 150 |
| 2026-09-05 | `240cc5b2` | sassy-loon-697 | 48.70% | 0.4913 | 80.99% | 3 | 0.03 | 150 |
| 2026-08-23 | `268b11cb` | respected-cub-847 | 51.94% | 0.5224 | 109.52% | 3 | 0.03 | 150 |
| 2026-08-23 | `2cb84ecf` | secretive-snail-371 | 50.55% | 0.5069 | 192.55% | 3 | 0.03 | 150 |
| 2026-03-12 | `2f8fabd1` | clumsy-snipe-158 | 50.74% | - | 152393.89% | 4 | 0.05 | 200 |
| 2026-03-13 | `3acfebb9` | gaudy-mink-321 | 51.79% | - | 315.54% | 4 | 0.05 | 200 |
| 2026-09-26 | `64e00bd3` | mercurial-ant-519 | 49.87% | 0.5030 | 18.70% | 3 | 0.03 | 150 |
| 2026-08-23 | `999ad0c8` | nebulous-bear-624 | 50.60% | 0.5017 | 221.87% | 3 | 0.03 | 150 |
| 2026-03-06 | `b7de8efc` | gregarious-cub-73 | 51.29% | - | 37.07% | 4 | 0.05 | 200 |

#### META (16 Recorded Runs)
| Date | Run ID | Run Name | Accuracy | ROC-AUC | Return | Max Depth | LR | N Est |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-09-26 | `077cb309` | zealous-newt-217 | 50.91% | 0.5095 | 152.27% | 3 | 0.03 | 150 |
| 2026-08-23 | `0cc853e8` | grandiose-gnu-703 | 51.24% | 0.5058 | 1.24% | 3 | 0.03 | 150 |
| 2026-10-01 | `2dbc21b3` | honorable-bass-934 | 50.91% | 0.5095 | 152.27% | 3 | 0.03 | 150 |
| 2026-09-20 | `39d3697c` | inquisitive-moth-727 | 51.43% | 0.5104 | -28.11% | 3 | 0.03 | 150 |
| 2026-03-12 | `3ff539ab` | aged-trout-280 | 49.35% | - | 492704.80% | 4 | 0.05 | 200 |
| 2026-03-12 | `4f36f86c` | gregarious-goat-217 | 49.35% | - | 492704.80% | 4 | 0.05 | 200 |
| 2026-03-13 | `75700142` | brawny-donkey-358 | 49.70% | - | 74.41% | 4 | 0.05 | 200 |
| 2026-09-10 | `9b2b36e8` | omniscient-chimp-334 | 50.91% | 0.5095 | 152.27% | 3 | 0.03 | 150 |
| 2026-08-23 | `a6a2e3ae` | tasteful-moth-966 | 50.74% | 0.4993 | 6.17% | 3 | 0.03 | 150 |
| 2026-03-06 | `bd957e8c` | kindly-toad-985 | 50.74% | - | 41.83% | 4 | 0.05 | 200 |

#### AMZN (14 Recorded Runs)
| Date | Run ID | Run Name | Accuracy | ROC-AUC | Return | Max Depth | LR | N Est |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-08-23 | `0d1f00a6` | abundant-kite-25 | 49.40% | 0.4974 | -14.33% | 3 | 0.03 | 150 |
| 2026-08-23 | `12beb74d` | chill-wasp-319 | 48.41% | 0.4889 | 41.04% | 3 | 0.03 | 150 |
| 2026-09-20 | `14ee01ec` | incongruous-swan-338 | 51.30% | 0.5025 | 60.19% | 3 | 0.03 | 150 |
| 2026-08-23 | `23071dbd` | persistent-bear-465 | 50.00% | 0.4989 | 21.85% | 3 | 0.03 | 150 |
| 2026-03-12 | `2e3f6aeb` | ambitious-shrike-54 | 50.74% | - | 52545.00% | 4 | 0.05 | 200 |
| 2026-03-14 | `455ff38d` | loud-perch-461 | 49.16% | - | 20491.32% | 4 | 0.05 | 200 |
| 2026-08-23 | `501e2a84` | calm-conch-951 | 49.40% | 0.4974 | -14.33% | 3 | 0.03 | 150 |
| 2026-09-26 | `759bd8a9` | hilarious-asp-154 | 50.85% | 0.5055 | -45.14% | 3 | 0.03 | 150 |
| 2026-09-05 | `90d92c61` | adaptable-gnat-899 | 51.11% | 0.4993 | -28.90% | 3 | 0.03 | 150 |
| 2026-08-23 | `9b57ad4d` | fortunate-hound-649 | 50.74% | 0.4990 | 12.92% | 4 | 0.05 | 200 |

#### TSLA (15 Recorded Runs)
| Date | Run ID | Run Name | Accuracy | ROC-AUC | Return | Max Depth | LR | N Est |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-08-23 | `29bcbc9f` | likeable-lark-385 | 50.00% | 0.4852 | 210.57% | 3 | 0.03 | 150 |
| 2026-03-14 | `2f488eb5` | unique-boar-759 | 49.90% | - | 120058.50% | 4 | 0.05 | 200 |
| 2026-09-10 | `3a96e688` | calm-squirrel-306 | 51.24% | 0.5047 | -51.39% | 3 | 0.03 | 150 |
| 2026-09-20 | `4fe90fb7` | angry-fawn-361 | 51.11% | 0.5035 | -71.01% | 3 | 0.03 | 150 |
| 2026-08-23 | `521c9355` | persistent-carp-856 | 48.41% | 0.4759 | 221.87% | 3 | 0.03 | 150 |
| 2026-09-26 | `5248d0e0` | righteous-bear-298 | 51.24% | 0.5047 | -51.39% | 3 | 0.03 | 150 |
| 2026-08-23 | `61d73895` | grandiose-donkey-663 | 50.00% | 0.4855 | -47.93% | 3 | 0.03 | 150 |
| 2026-03-06 | `7d4fecf6` | wise-whale-287 | 49.01% | - | -15.96% | 4 | 0.05 | 200 |
| 2026-03-13 | `847c83a5` | exultant-hare-35 | 49.90% | - | 523.86% | 4 | 0.05 | 200 |
| 2026-10-01 | `90f079e9` | invincible-hare-539 | 51.24% | 0.5047 | -51.39% | 3 | 0.03 | 150 |

---

## 3. Top 25 Highest Accuracy Models Across Full Universe

| Rank | Ticker | Date | Accuracy | ROC-AUC | Return | Run Name | Run ID |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **SBXE** | 2026-09-22 | **100.00%** | nan | 0.00% | gaudy-snipe-183 | `02855dbe` |
| 2 | **BEBE** | 2026-09-22 | **100.00%** | nan | 0.00% | defiant-gnat-206 | `0db94832` |
| 3 | **PLTS** | 2026-09-22 | **100.00%** | nan | 0.00% | able-goat-108 | `1248f2a1` |
| 4 | **ISNR** | 2026-09-22 | **100.00%** | nan | 0.00% | glamorous-slug-738 | `1353b368` |
| 5 | **ITHA** | 2026-09-22 | **100.00%** | nan | 0.00% | dazzling-kit-882 | `1de41a21` |
| 6 | **ZKPU** | 2026-09-22 | **100.00%** | nan | 0.00% | salty-goat-168 | `25b7accd` |
| 7 | **TRGS** | 2026-09-22 | **100.00%** | nan | 0.00% | zealous-mole-346 | `291f8934` |
| 8 | **TP** | 2026-09-22 | **100.00%** | nan | 0.00% | industrious-hawk-140 | `304205e2` |
| 9 | **CAQ** | 2026-09-22 | **100.00%** | nan | 0.00% | luminous-squid-724 | `34b0eabf` |
| 10 | **IACQ** | 2026-09-22 | **100.00%** | nan | 0.00% | luxuriant-robin-560 | `3f0c8674` |
| 11 | **EFTY** | 2026-09-22 | **100.00%** | nan | 0.00% | agreeable-bee-301 | `3f5cb7de` |
| 12 | **MCTA** | 2026-09-22 | **100.00%** | nan | 0.00% | selective-goat-998 | `499baf26` |
| 13 | **NEWTO** | 2026-09-22 | **100.00%** | nan | 0.00% | redolent-hog-901 | `4b6e78f7` |
| 14 | **GD** | 2026-09-05 | **100.00%** | nan | 0.00% | bald-stork-591 | `7c493ec7` |
| 15 | **RACD** | 2026-09-22 | **100.00%** | nan | 0.00% | flawless-koi-973 | `7c7ceeb8` |
| 16 | **LPCV** | 2026-09-22 | **100.00%** | nan | 0.00% | big-crane-9 | `920c0311` |
| 17 | **ATLQ** | 2026-09-22 | **100.00%** | nan | 0.00% | adaptable-toad-404 | `94cae1e8` |
| 18 | **WLII** | 2026-09-22 | **100.00%** | nan | 0.00% | monumental-gull-924 | `a6f7c7ca` |
| 19 | **ACGC** | 2026-09-21 | **100.00%** | nan | 0.00% | smiling-lamb-946 | `a74c727a` |
| 20 | **ALOV** | 2026-09-22 | **100.00%** | nan | 0.00% | awesome-hare-561 | `a87a90f0` |
| 21 | **APMC** | 2026-09-22 | **100.00%** | nan | 0.00% | glamorous-gnu-778 | `c8721893` |
| 22 | **VCRE** | 2026-09-22 | **100.00%** | nan | 0.00% | serious-crow-645 | `ccccae78` |
| 23 | **IRAB** | 2026-09-22 | **100.00%** | nan | 0.00% | traveling-crow-222 | `d3417efa` |
| 24 | **NHIV** | 2026-09-22 | **100.00%** | nan | 0.00% | calm-hen-332 | `e071b80b` |
| 25 | **BKYI** | 2026-09-22 | **100.00%** | nan | 0.00% | nervous-stoat-415 | `e2a8d2ca` |

---

## 4. Top 25 Highest Alpha / Total Return Models Across Full Universe

| Rank | Ticker | Date | Strategy Return | B&H Return | Accuracy | Run Name | Run ID |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **META** | 2026-03-14 | **572043.41%** | 255.28% | 49.70% | adaptable-elk-178 | `ee9ac572` |
| 2 | **META** | 2026-03-12 | **492704.80%** | 255.28% | 49.35% | gregarious-goat-217 | `4f36f86c` |
| 3 | **META** | 2026-03-12 | **492704.80%** | 255.28% | 49.35% | aged-trout-280 | `3ff539ab` |
| 4 | **MSFT** | 2026-03-14 | **403240.08%** | 365.25% | 49.35% | funny-grouse-109 | `4d1fa0be` |
| 5 | **GOOGL** | 2026-03-12 | **152393.89%** | 439.75% | 50.74% | victorious-bee-822 | `ca96a62a` |
| 6 | **GOOGL** | 2026-03-12 | **152393.89%** | 439.75% | 50.74% | clumsy-snipe-158 | `2f8fabd1` |
| 7 | **TSLA** | 2026-03-14 | **120058.50%** | 1704.54% | 49.90% | unique-boar-759 | `2f488eb5` |
| 8 | **GOOGL** | 2026-03-14 | **119462.40%** | 439.75% | 51.79% | mysterious-fawn-994 | `d471da20` |
| 9 | **DXF** | 2026-09-22 | **89515.39%** | -92.66% | 61.59% | fearless-wren-604 | `890eda62` |
| 10 | **TSLA** | 2026-03-12 | **67503.18%** | 1704.54% | 48.21% | delightful-snail-882 | `b85aca66` |
| 11 | **AMZN** | 2026-03-12 | **52545.00%** | 169.87% | 50.74% | ambitious-shrike-54 | `2e3f6aeb` |
| 12 | **NVDA** | 2026-03-14 | **42020.09%** | 2921.25% | 50.12% | delightful-dove-470 | `4290d669` |
| 13 | **NVDA** | 2026-03-12 | **39342.74%** | 2953.30% | 49.75% | silent-shark-516 | `378a3a7b` |
| 14 | **NVDA** | 2026-03-12 | **39342.74%** | 2953.30% | 49.75% | whimsical-squid-331 | `5b5b5369` |
| 15 | **NVDA** | 2026-03-12 | **39342.74%** | 2953.30% | 49.75% | enthused-fowl-153 | `fc7f256c` |
| 16 | **NVDA** | 2026-03-14 | **37841.90%** | 2953.30% | 50.20% | delicate-quail-441 | `4240a784` |
| 17 | **AAPL** | 2026-03-14 | **34514.17%** | 514.25% | 48.61% | peaceful-dolphin-514 | `21341a5d` |
| 18 | **MSFT** | 2026-03-12 | **33011.54%** | 365.25% | 49.85% | gaudy-mink-676 | `22453673` |
| 19 | **MSFT** | 2026-03-12 | **33011.54%** | 365.25% | 49.85% | funny-snail-937 | `61b7d8d1` |
| 20 | **AAPL** | 2026-03-12 | **28191.26%** | 514.25% | 47.47% | rogue-goat-322 | `1556bf0c` |
| 21 | **AAPL** | 2026-03-12 | **28191.26%** | 514.25% | 47.47% | mercurial-panda-692 | `7af16c8c` |
| 22 | **VAL** | 2026-09-22 | **24881.30%** | 12821.48% | 51.02% | vaunted-zebra-635 | `c457104c` |
| 23 | **AMZN** | 2026-03-14 | **20491.32%** | 169.87% | 49.16% | loud-perch-461 | `455ff38d` |
| 24 | **PVLA** | 2026-09-22 | **19920.62%** | -30.78% | 49.16% | grandiose-tern-392 | `0a7a6c50` |
| 25 | **GPOR** | 2026-09-22 | **18536.18%** | 14300.93% | 51.72% | trusting-bee-107 | `1e174dde` |

---

## 5. Retention & Storage Policy

- **Preserved Locally in `mlruns/`**: All active recent runs (post-2026-09-25) and top benchmark runs.
- **Preserved Permanently**: Complete metadata, metric tables, and parameter specifications in `docs/mlflow_runs_archive.csv`.
- **Production Models**: Active deployment models reside in `models/*.json` (unaffected by MLflow pruning).
- **Live Portfolio & Ledger**: Strictly preserved in `results/paper_portfolio.json` and `results/executed_trades.csv`.
