# CQROS Risk Policy

Version: 1.0.0

Status: Active

Owner: Risk Management Layer

---

# 1. Purpose

This document defines the official risk management policy
for CQROS.

Every trading decision must satisfy these rules before
execution.

Risk controls always override strategy signals.

---

# 2. Objectives

The CQROS risk framework aims to

- Preserve capital
- Limit downside
- Control portfolio exposure
- Prevent catastrophic losses
- Ensure long-term survivability
- Maintain consistent risk-adjusted returns

---

# 3. Risk Hierarchy

```
Exchange Rules

↓

Account Limits

↓

Portfolio Limits

↓

Strategy Limits

↓

Position Limits

↓

Order Validation
```

Higher-level rules always take precedence.

---

# 4. General Principles

CQROS follows

- Capital preservation first
- Risk before return
- Diversification
- Controlled leverage
- Deterministic sizing
- Automated protection

---

# 5. Position Sizing

Position sizing is determined using

- Portfolio value
- Maximum account risk
- Stop-loss distance
- Instrument volatility
- Liquidity
- Correlation

Supported methods

- Fixed percentage
- Volatility adjusted
- ATR based
- Kelly Fraction (research only)
- Equal risk contribution

---

# 6. Maximum Position Risk

Default

```
Maximum risk per trade

1.0%

of portfolio equity
```

Hard maximum

```
2.0%
```

No strategy may exceed this limit.

---

# 7. Portfolio Risk

Maximum total portfolio risk

```
5%

of portfolio equity
```

New positions are rejected when exceeded.

---

# 8. Maximum Positions

Default

```
10
```

Configurable.

Risk engine may reduce this dynamically.

---

# 9. Portfolio Exposure

Maximum directional exposure

Long

70%

Short

70%

Net exposure limits are configurable.

---

# 10. Correlation Control

Avoid concentration.

Default

Maximum exposure to highly correlated assets

30%

Example

BTC

ETH

SOL

may belong to the same correlation cluster.

---

# 11. Volatility Scaling

Position size decreases as volatility increases.

Preferred volatility measure

ATR

Alternative

Realized Volatility

---

# 12. Liquidity Filter

Reject trades when

- Spread exceeds threshold
- Daily volume below minimum
- Order book depth insufficient
- Slippage estimate exceeds limit

---

# 13. Stop Loss Policy

Every position requires

- Initial stop-loss
- Stored before order submission

No exceptions.

---

# 14. Take Profit Policy

Supported methods

- Fixed Risk:Reward
- ATR Multiple
- Trailing Stop
- Dynamic Exit
- Strategy Controlled

---

# 15. Trailing Stop

Optional.

Supported

- ATR trailing
- Percentage trailing
- Swing structure trailing

---

# 16. Partial Exits

Supported

Examples

- 25%
- 50%
- 75%

Partial exits update remaining position risk.

---

# 17. Break-Even Logic

Supported.

Move stop-loss to entry after configurable profit.

Default

```
1R
```

---

# 18. Maximum Drawdown

Soft limit

10%

Hard limit

15%

Hard limit triggers

Trading suspension.

---

# 19. Daily Loss Limit

Trading stops when

Daily realized loss exceeds

```
3%

of account equity.
```

Trading resumes next session.

---

# 20. Consecutive Loss Protection

Default

Maximum consecutive losses

5

After reaching the threshold

- Pause trading
- Notify operator
- Require manual review (live mode)

---

# 21. Leverage

Research

Unlimited simulation

Paper

Configurable

Production

Exchange limits

Internal leverage cap may be lower.

---

# 22. Slippage Protection

Reject trades when expected slippage exceeds

Configured threshold.

---

# 23. Spread Protection

Reject trades when

Bid-ask spread exceeds maximum.

---

# 24. Funding Protection

For perpetual futures

Reduce exposure when funding becomes extreme.

Funding limits are configurable.

---

# 25. Open Interest Protection

Large abnormal OI changes

↓

Increase caution

↓

Reduce position size

---

# 26. Volatility Circuit Breaker

Suspend new entries when

Volatility exceeds configured threshold.

Existing positions continue to be managed.

---

# 27. Market Halt Detection

Stop trading when

- Exchange unavailable
- API degraded
- Trading halted
- Symbol suspended

---

# 28. Model Risk

Reject signals when

- Prediction confidence below threshold
- Model expired
- Feature drift detected
- Data validation fails

---

# 29. Data Quality Protection

Trading prohibited when

- Missing candles
- Timestamp mismatch
- Invalid prices
- Corrupted features

---

# 30. Portfolio Concentration

Maximum allocation per asset

```
15%
```

Default.

---

# 31. Order Validation

Before submission verify

- Quantity
- Tick size
- Step size
- Minimum notional
- Margin availability
- Position limits

---

# 32. Emergency Shutdown

Immediately disable new trading when

- Exchange instability
- Data corruption
- Database unavailable
- Risk engine failure
- Critical software error

Existing positions remain under protective management
whenever possible.

---

# 33. Recovery

After emergency shutdown

Verify

- Data integrity
- Connectivity
- Exchange health
- Portfolio state

Only then resume trading.

---

# 34. Logging

Every risk decision records

- Timestamp
- Rule triggered
- Position
- Portfolio state
- Action taken

Audit logs are immutable.

---

# 35. Monitoring

Track continuously

- Drawdown
- Exposure
- Leverage
- Margin
- Volatility
- Win/Loss streak
- Portfolio concentration

---

# 36. Stress Testing

Validate against

- Flash crashes
- High volatility
- Exchange outages
- Network failures
- Liquidity collapse
- Extreme slippage

---

# 37. Backtesting

Every risk rule must behave identically in

- Research
- Backtesting
- Paper Trading
- Live Trading

No environment-specific logic unless explicitly documented.

---

# 38. Configuration

All thresholds are configurable.

Changes require

- Validation
- Versioning
- Audit logging

---

# 39. Review Process

Risk policy is reviewed

- Before major releases
- After significant drawdowns
- After production incidents
- During annual architecture reviews

---

# 40. Summary

The CQROS Risk Policy establishes a layered,
deterministic framework that prioritizes capital
preservation over opportunity.

Every signal, order, and position must pass these controls
before execution, ensuring consistent behavior across
research, backtesting, paper trading, and live trading
while maintaining institutional-grade risk governance.