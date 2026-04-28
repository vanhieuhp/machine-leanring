# Time Series Forecasting - Learning Guide

## What is Time Series Forecasting?

Time series forecasting predicts future values based on historical data points collected at regular intervals.

## Why Time Series Matters

- **Financial forecasting**: Stock prices, cryptocurrency, market trends
- **Business planning**: Demand forecasting, sales predictions
- **Operations**: Inventory management, resource allocation
- **Science**: Weather forecasting, climate modeling

## Learning Objectives

By the end of this section, you'll master:

### Statistical Methods
1. **ARIMA** - Autoregressive Integrated Moving Average
2. **SARIMA** - Seasonal ARIMA
3. **Exponential Smoothing** - Simple, Holt, Holt-Winters

### Deep Learning Methods
1. **LSTM** - Long Short-Term Memory networks
2. **GRU** - Gated Recurrent Units
3. **Transformer** - Attention-based models

### Modern Tools
1. **Prophet** - Facebook's forecasting tool
2. **NeuralProphet** - Neural network-based Prophet

## Key Concepts

### 1. Time Series Components

```
┌─────────────────────────────────────────────────────────┐
│            TIME SERIES DECOMPOSITION                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Original:    ┌─────────────────────────────┐          │
│               │ ╱╲    ╱╲    ╱╲    ╱╲    ╱  │          │
│               │╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲╳         │
│               └─────────────────────────────┘          │
│                                                          │
│  Trend:     ────────────────────────────────           │
│               (Long-term direction)                     │
│                                                          │
│  Seasonal:  ~~~~~  ~~~~~  ~~~~~  ~~~~~                │
│               (Repeating pattern)                       │
│                                                          │
│  Residual:   │  │  │  │  │  │  │  │  │  │             │
│               (Random noise)                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2. Stationarity

**Stationary series**: Constant mean, variance over time
- Required for ARIMA
- Can be achieved through differencing

```
Non-Stationary → Stationary
     ↓
  Differencing
     ↓
  d = 1 or 2
```

### 3. ACF and PACF

| Function | What it shows |
|----------|---------------|
| **ACF** (Autocorrelation) | Correlation between series and its lags |
| **PACF** (Partial ACF) | Direct correlation with lags (excluding indirect) |

### 4. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root MSE |
| **MAPE** | Mean Absolute Percentage Error |
| **MASE** | Mean Absolute Scaled Error |

## Study Path

### Week 1: Statistical Methods
1. **Start with**: Statistical fundamentals
   - Understanding trends, seasonality
   - Stationarity tests (ADF, KPSS)
   - ACF/PACF analysis

2. **Then**: ARIMA
   - AR (p), I (d), MA (q) components
   - Grid search for parameters
   - Model diagnostics

### Week 2: Exponential Smoothing
3. **Next**: Exponential Smoothing
   - Simple exponential smoothing
   - Holt's linear trend
   - Holt-Winters (additive/multiplicative)

### Week 3: Deep Learning
4. **Then**: LSTM/GRU
   - Sequence preparation
   - Window-based training
   - Architecture design

### Week 4: Modern Tools & Projects
5. **Finally**: Prophet & Projects
   - Prophet for quick forecasting
   - Stock price prediction project
   - Compare all methods

## Common Mistakes to Avoid

1. **Ignoring stationarity** - Always check and transform
2. **Data leakage** - No future information in training
3. **Not handling seasonality** - Critical for many domains
4. **Overfitting** - Validate on holdout period
5. **Ignoring external factors** - Consider holidays, events

## Tips for Success

1. **Start simple** - Baseline with naive methods
2. **Visualize** - Always plot your data
3. **Decompose** - Understand components
4. **Validate properly** - Time series cross-validation
5. **Ensemble** - Combine multiple models

---

**Difficulty**: Expert

**Estimated Time**: 1 month

**Prerequisites**: Phase 1 (Statistics, NumPy, Pandas)

**Next**: Computer Vision (Month 3)
