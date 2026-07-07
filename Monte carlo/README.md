# Monte Carlo Portfolio Risk & Strategy Simulator

A Monte Carlo-based portfolio risk simulation engine for a 5-asset tech portfolio (MU, SNDK, NVDA, ARM, NOW), comparing multiple modeling assumptions and evaluating a machine learning trading signal.

## What this project does

- Simulates 1,000+ correlated price paths using Geometric Brownian Motion (GBM), calibrated from historical price data.
- Computes portfolio-level and per-asset risk metrics: **Value at Risk (VaR)**, **Conditional Value at Risk (CVaR)**, and **maximum drawdown**.
- Diagnoses and corrects a drift-estimation bias, comparing raw historical drift against a market-benchmarked, mean-reverting alternative.
- Implements and compares four distinct simulation methodologies:
  - Standard GBM (raw historical drift)
  - Mean-reverting GBM (shrunk drift)
  - i.i.d. bootstrap resampling
  - Block bootstrap resampling (preserves volatility clustering)
- Validates findings with a 10-seed robustness check to confirm differences between methods are structural, not random noise.
- Builds a logistic regression classifier on technical indicators to generate a trading signal, and uses it to tilt portfolio weights — critically evaluated against equal-weighting rather than assumed to add value.

## Key findings

- Naive historical drift, when calibrated on stocks mid-supercycle (MU, SNDK both had extraordinary real-world rallies), produces implausible expected returns. Correcting for this cuts expected portfolio value by roughly two-thirds.
- Drift and volatility are independent risk levers — lowering expected return does not by itself reduce drawdown risk.
- GBM's normal-distribution assumption understates real tail risk relative to resampling actual historical returns.
- How you resample matters as much as whether you do: i.i.d. bootstrap understates drawdown risk by breaking up real crash sequences; block bootstrap partially restores it.
- A simple ML trading signal (49-59% test accuracy) did not produce a meaningful edge over equal-weighting — a result consistent with weak-form market efficiency.

## Tech stack

- Python, NumPy, pandas, matplotlib
- yfinance (historical price data)
- scikit-learn (logistic regression classifier)

## Setup

```bash
pip install -r requirements.txt
jupyter notebook "Monte_Carlo_Portfolio_Risk___Strategy_Simulator.ipynb"
```

## Structure

The notebook is organized into sections: data loading → GBM simulation engine → portfolio-level risk metrics → drift sensitivity analysis → bootstrap methods → robustness validation → ML trading signal → conclusion. A glossary of terms is included at the end for readers unfamiliar with quant finance terminology.

## Disclaimer

This project is for educational purposes only and does not constitute financial advice. Historical stock performance is not indicative of future results.
