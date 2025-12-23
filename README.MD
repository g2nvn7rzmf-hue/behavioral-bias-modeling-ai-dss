# 🧠 Behavioral Bias Modeling in AI-Based Decision Support Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Paper](https://img.shields.io/badge/Paper-Published-green.svg)](#citation)

## 📋 Overview

This repository contains the complete implementation of the Monte Carlo simulation framework described in the research paper:

> **"Integrating Behavioral Bias Modeling into AI-Based Decision Support Systems for Sustainable Investment Planning"**

The framework explicitly integrates behavioral bias modeling into AI-based financial decision support systems, comparing investment outcomes under purely rational, data-driven DSS versus enhanced DSS that accounts for behavioral distortions.

## 🎯 Key Features

- **Three Behavioral Biases Modeled:**
  - 🔴 **Loss Aversion** (λ=2.0): Asymmetric weighting of losses vs. gains
  - 🟡 **Overconfidence** (β=0.35): Systematic underestimation of risk
  - 🟢 **Herding Behavior** (γ=0.25): Consensus-based decision adjustment

- **Comprehensive Simulation Framework:**
  - N=5,000 Monte Carlo iterations
  - M=30 investment alternatives per scenario
  - 7 ablation study conditions
  - Bootstrap confidence intervals (10,000 iterations)

- **Performance Metrics:**
  - Mean realized return
  - Return volatility (σ)
  - Value at Risk (VaR 5%)
  - Conditional VaR (CVaR 5%)
  - Decision regret
  - Decision stability

## 📊 Key Results (from Paper)

| Metric | Rational DSS | Bias-Aware DSS | Change |
|--------|--------------|----------------|--------|
| Mean Return | 0.158 | 0.146 | -7.2% |
| Volatility | 0.211 | 0.189 | **-10.6%** |
| VaR (5%) | -0.181 | -0.154 | **-15.1%** |
| CVaR (5%) | -0.267 | -0.238 | **-10.8%** |
| Decision Stability | 0.491 | 0.816 | **+66.2%** |


## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Install required packages
pip install numpy pandas scipy matplotlib seaborn tqdm
```

### Installation

```bash
# Clone or download the repository
cd /Users/abtin/Desktop/maqaleeee

# Verify installation
python -c "from behavioral_bias_simulation_final import *; print('✅ Installation successful!')"
```

### Running Quick Test

```bash
# Run quick test with reduced iterations (N=1000)
python quick_test.py
```

### Running Full Simulation

```python
from behavioral_bias_simulation_final import (
    SimulationParameters,
    BiasParameters,
    MonteCarloSimulation,
    AblationStudy
)

# Full simulation with paper parameters
sim_params = SimulationParameters(
    n_simulations=5000,
    n_alternatives=30
)

# Run ablation study
ablation = AblationStudy(sim_params)
results = ablation.run_all_conditions()

# Print results
for condition, data in results.items():
    print(f"{condition}: {data['metrics']}")
```

## 📐 Mathematical Formulations

### 1. Loss Aversion (Section 3.2.1)

```
V(μᵢ) = μᵢ           if μᵢ ≥ r (reference point)
V(μᵢ) = λ · μᵢ       if μᵢ < r

where λ = 2.0 (loss aversion coefficient)
```

### 2. Overconfidence (Section 3.2.2)

```
σ̂ᵢ = σᵢ / (1 + β)

where β = 0.35 (overconfidence intensity)
```

### 3. Herding Behavior (Section 3.2.3)

```
Sᵢʳ = (1 - γ) · Sᵢ + γ · Cᵢ

where γ = 0.25 (herding intensity)
      Cᵢ = consensus score
```

### 4. Decision Score

```
Sᵢ = w₁ · V(μᵢ) - w₂ · σᵢ + w₃ · ESGᵢ

where w₁ = 0.50, w₂ = 0.30, w₃ = 0.20
```

## 🔬 Ablation Study Conditions

| Condition | λ | β | γ | Purpose |
|-----------|---|---|---|---------|
| Baseline (Rational) | 1.0 | 0.0 | 0.0 | Control group |
| Loss Aversion Only | 2.0 | 0.0 | 0.0 | Isolate LA effect |
| Overconfidence Only | 1.0 | 0.35 | 0.0 | Isolate OC effect |
| Herding Only | 1.0 | 0.0 | 0.25 | Isolate HB effect |
| Combined Biases | 2.0 | 0.35 | 0.25 | Full model |
| Low Bias Intensity | 1.5 | 0.20 | 0.15 | Sensitivity test |
| High Bias Intensity | 2.5 | 0.50 | 0.35 | Sensitivity test |

## 📈 Data Distributions

| Parameter | Distribution | Values |
|-----------|--------------|--------|
| Expected Return (μᵢ) | Normal | N(0.08, 0.04) |
| Volatility (σᵢ) | LogNormal | LN(-2.3, 0.5) |
| ESG Score | Beta | Beta(2, 5) |
| Return Forecast Error | Normal | N(0, 0.03) |
| Volatility Forecast Error | Normal | N(0, 0.18) |

## 🛠️ API Reference

### SimulationParameters

```python
@dataclass
class SimulationParameters:
    n_simulations: int = 5000          # Number of Monte Carlo iterations
    n_alternatives: int = 30           # Investment alternatives per scenario
    mu_mean: float = 0.08              # Mean of expected returns
    mu_std: float = 0.04               # Std of expected returns
    sigma_loc: float = -2.3            # LogNormal location for volatility
    sigma_scale: float = 0.5           # LogNormal scale for volatility
    esg_alpha: float = 2.0             # Beta distribution alpha
    esg_beta: float = 5.0              # Beta distribution beta
    forecast_error_std: float = 0.03   # Return forecast error
    sigma_forecast_error_std: float = 0.18  # Volatility forecast error
    var_confidence: float = 0.05       # VaR confidence level
    n_bootstrap: int = 10000           # Bootstrap iterations
    w_return: float = 0.50             # Weight for return
    w_risk: float = 0.30               # Weight for risk
    w_esg: float = 0.20                # Weight for ESG
```

### BiasParameters

```python
@dataclass
class BiasParameters:
    lambda_loss_aversion: float = 2.0   # Loss aversion coefficient
    beta_overconfidence: float = 0.35   # Overconfidence intensity
    gamma_herding: float = 0.25         # Herding intensity
```

### Key Classes

| Class | Description |
|-------|-------------|
| `InvestmentAlternative` | Single investment with true/forecasted parameters |
| `BehavioralBiasModel` | Implements three bias transformations |
| `DecisionSupportSystem` | AI-based DSS with bias modeling |
| `PerformanceMetrics` | Compute VaR, CVaR, stability, regret |
| `MonteCarloSimulation` | Main simulation framework |
| `AblationStudy` | Run all 7 experimental conditions |

## 📊 Output Example

```
============================================================
Ablation Study Results
============================================================
Condition              Return   Volatility   VaR      Stability
------------------------------------------------------------
Baseline (Rational)    0.158    0.211       -0.181    0.491
Loss Aversion Only     0.149    0.198       -0.166    0.687
Overconfidence Only    0.164    0.237       -0.189    0.413
Herding Only           0.155    0.203       -0.176    0.621
Combined Biases        0.146    0.189       -0.154    0.816
============================================================
```

## ⚠️ Known Issues & Fixes

See [fix.md](fix.md) for documented issues and their solutions.

### Fixed Issues:
- ✅ Realized returns computed once for all alternatives
- ✅ gamma_herding parameter added to BiasParameters
- ✅ Consistent random seed for reproducibility

## 🧪 Testing

```bash
# Quick test (N=1000, ~30 seconds)
python quick_test.py

# Full test (N=5000, ~2-3 minutes)
python behavioral_bias_simulation_final.py

# Individual bias test
python -c "
from behavioral_bias_simulation_final import *
sim = MonteCarloSimulation(
    SimulationParameters(n_simulations=500),
    BiasParameters(lambda_loss_aversion=2.0, beta_overconfidence=0.0, gamma_herding=0.0),
    {'loss_aversion': True, 'overconfidence': False, 'herding': False}
)
results = sim.run_full_simulation()
print(sim.compute_aggregate_metrics(results))
"
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{behavioral_bias_dss_2025,
  title={Integrating Behavioral Bias Modeling into AI-Based Decision 
         Support Systems for Sustainable Investment Planning},
  author={Research Team},
  journal={Journal Name},
  year={2025},
  keywords={Artificial Intelligence, Decision Support Systems, 
            Behavioral Bias Modeling, Sustainable Investment}
}
```

## 📖 References

1. Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263-291.
2. Petersen, K. (2023). Overconfidence bias and its impact on financial decision-making. *Journal of Behavioral Finance*, 24(2), 187-201.
3. Shah, S., Zhang, T., & Zhang, Y. (2019). Does herding bias drive the firm value? *Sustainability*, 11(20), 5583.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Research Team
- Contact: [your-email@example.com]

## 🙏 Acknowledgments

- This research received no specific grant from funding agencies
- Thanks to industry practitioners for valuable discussions

---

**⭐ If you find this useful, please star the repository!**