## Trader Performance vs Market Sentiment (Fear/Greed)

---
## 📁 Project Structure

```
primetrade_project/
│
├── trader_sentiment_analysis.ipynb   ← Main analysis notebook (run this)
├── analysis.py                        ← Pure Python version of the same pipeline
├── README.md                          ← This file
├── writeup.md                         ← 1-page methodology + insights summary
│
├── data/
│   ├── fear_greed_index.csv           ← Raw Fear/Greed dataset (place here)
│   ├── historical_data.csv            ← Raw Hyperliquid trader data (place here)
│   └── merged_daily.csv               ← Auto-generated merged dataset
│
└── charts/
    ├── chart1_performance_by_sentiment.png
    ├── chart2_behavior_by_sentiment.png
    ├── chart3_segment_by_sentiment.png
    ├── chart4_freq_segment.png
    ├── chart5_longratio_heatmap.png
    ├── chart6_fg_vs_pnl_scatter.png
    └── chart7_model_results.png        ← Bonus: predictive model
```

---

## ⚙️ Setup & How to Run

### Requirements
```
Python >= 3.9
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost
jupyter
```

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### Place data files
Put both raw CSVs in the `data/` folder:
- `data/fear_greed_index.csv`
- `data/historical_data.csv`

### Run the notebook
```bash
jupyter notebook trader_sentiment_analysis.ipynb
```
Run all cells top-to-bottom (Kernel → Restart & Run All).

### Run the Streamlit app
```bash
pip install streamlit
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## 📊 What's Inside

| Part | Description |
|------|-------------|
| **A** | Data loading, cleaning, deduplication, timestamp alignment, feature engineering |
| **B** | 6 charts analyzing performance & behavior by sentiment, trader segmentation |
| **C** | 3 actionable strategy recommendations with evidence |
| **Bonus** | Gradient Boosting classifier predicting next-day profitability (67.3% accuracy) |

---

# Write-Up: Trader Performance vs Market Sentiment

---

## Methodology

**Datasets:** Bitcoin Fear/Greed Index (2,644 daily observations, 2018–2025) merged with Hyperliquid historical trader data (211,224 trades, 32 unique accounts, May 2023–May 2025).

**Cleaning steps:**
- Removed 6,372 duplicate trades (matched on Account + Timestamp + Trade ID)
- Parsed `Timestamp IST` (DD-MM-YYYY HH:MM format) and extracted date for daily alignment
- Zero missing values in either dataset after cleaning
- Aligned by date (inner join → 2,340 account-day observations across 479 unique days)

**Features engineered:**
- Daily PnL per account, win rate, trade count, long/short ratio, average trade size, total volume
- Drawdown proxy: sum of negative PnL days per account
- Trader segments: High/Low size, Frequent/Infrequent, Consistent Winners vs Others
- For the model: 3-day rolling averages (lagged), lag-1 sentiment value, lag-1 behavioral features

---

## Insights

**Insight 1 — Fear Days Are High-Risk, Not Low-Return**
Fear days (FG < 40) show the highest average daily PnL ($4,942) but also the highest average drawdown ($1,075 vs $891 on Greed days — a 20% premium). This is driven by a small cohort of traders exploiting volatility while the majority overtrades and incurs losses. The distribution is significantly wider (higher variance) on Fear days.

**Insight 2 — Traders Overtrade and Over-Long on Fear Days**
On Fear days, traders execute ~34% more trades and use ~43% larger position sizes than on Greed days. Their long ratio spikes to 52% vs 47% on Greed days — suggesting systematic "buy the dip" behavior. This panic-buying in downturns likely explains the elevated drawdown. The behavior is consistent across both High-Size and Low-Size trader segments (confirmed in heatmap).

**Insight 3 — Consistent Winners Behave Differently**
Only 3 of 32 accounts qualify as Consistent Winners (positive total PnL + >50% win rate). This segment has a win rate of 63.7% vs 37.7% for Others, and achieves this with fewer trades (disciplined, not overtrading). Their PnL advantage is most pronounced on Greed days, where they show both higher win rates and lower drawdown — suggesting they correctly reduce aggression on Fear days.

---

## Strategy Recommendations

**Strategy 1 — Fear-Day Capital Preservation Rule**
*When FG < 40:* Reduce position sizes ≥40%, cap daily trade count at 50% of 7-day rolling average, and avoid adding to losing longs. Evidence: Fear days carry 20% higher drawdown and 34% more trades — a combination that destroys edge for all but the top 3 accounts.

**Strategy 2 — Greed-Day Momentum with Long Bias**
*When FG > 60:* Scale up long exposure moderately, increase trade frequency by 20–30% for Frequent traders, and only size up if rolling 3-day PnL is positive. Evidence: Greed days show the best win rate (36.3%), lowest drawdown, and the Consistent Winners segment performs best here.

**Strategy 3 — Neutral Day Rebalancing**
*When FG 40–60:* Treat as a reset window. Lowest avg PnL ($3,212) and lowest win rate (35.5%) make this a poor time for directional aggression. Use it to review positions and rebalance toward 50/50 long-short.

---

## Bonus: Predictive Model

A Gradient Boosting classifier predicts next-day profitability bucket (Loss / Neutral / Profit) with **67.3% test accuracy** and **54.8% ± 5.3% 5-fold CV accuracy** (vs 33% random baseline on 3-class). Top features: 3-day rolling PnL (momentum), FG index value, and rolling win rate. The model performs best on Profit prediction (precision 69%, recall 90%) — practically useful for "should I trade aggressively tomorrow?" decisions. Loss detection is weaker due to class imbalance (only 9% of observations).


## 🔑 Key Findings (Quick Summary)

1. **Fear days = higher volatility, not necessarily lower returns** — avg PnL is actually higher on Fear days but so is drawdown (+20% vs Greed)
2. **Traders overtrade on Fear days** — 34% more trades, 43% larger sizes, stronger long bias (52%)
3. **Consistent Winners** (only 3 of 32 accounts) maintain >50% win rate and positive PnL — their behavior is much more disciplined on Greed days
4. **Best predictors for next-day profit**: recent 3-day PnL momentum, FG index value, rolling win rate

