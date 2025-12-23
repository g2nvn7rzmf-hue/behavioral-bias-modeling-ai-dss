"""
Quick Test Example - Behavioral Bias Simulation
This is a simplified version for quick testing with fewer iterations

Version 1.1 - Uses the bug-fixed behavioral_bias_simulation_final.py
"""

import numpy as np
import pandas as pd
from behavioral_bias_simulation_final import (
    SimulationParameters,
    BiasParameters,
    MonteCarloSimulation,
    AblationStudy
)

def quick_test():
    """
    Run a quick test with reduced iterations
    """
    print("="*60)
    print("Quick Test - Behavioral Bias Simulation")
    print("="*60)
    
    # Use fewer iterations for quick testing
    quick_params = SimulationParameters(
        n_simulations=1000,  # Reduced from 5000
        n_alternatives=30
    )
    
    print(f"\nRunning quick test with {quick_params.n_simulations} iterations")
    print("(Full paper uses 5000 iterations)\n")
    
    # Test 1: Baseline (Rational DSS)
    print("Test 1: Baseline Rational DSS")
    print("-" * 40)
    
    baseline_sim = MonteCarloSimulation(
        quick_params,
        bias_params=BiasParameters(
            lambda_loss_aversion=1.0,
            beta_overconfidence=0.0,
            gamma_herding=0.0
        ),
        apply_biases={
            'loss_aversion': False,
            'overconfidence': False,
            'herding': False
        }
    )
    
    baseline_results = baseline_sim.run_full_simulation()
    baseline_metrics = baseline_sim.compute_aggregate_metrics(baseline_results)
    
    print("\nBaseline Results:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test 2: Bias-Aware DSS (Combined Biases)
    print("\n\nTest 2: Bias-Aware DSS (All Biases)")
    print("-" * 40)
    
    bias_aware_sim = MonteCarloSimulation(
        quick_params,
        bias_params=BiasParameters(
            lambda_loss_aversion=2.0,
            beta_overconfidence=0.35,
            gamma_herding=0.25
        ),
        apply_biases={
            'loss_aversion': True,
            'overconfidence': True,
            'herding': True
        }
    )
    
    bias_aware_results = bias_aware_sim.run_full_simulation()
    bias_aware_metrics = bias_aware_sim.compute_aggregate_metrics(bias_aware_results)
    
    print("\nBias-Aware Results:")
    for metric, value in bias_aware_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Compute percentage changes
    print("\n\nPercentage Changes (Bias-Aware vs. Baseline):")
    print("-" * 40)
    
    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        bias_aware_val = bias_aware_metrics[metric]
        
        if baseline_val != 0:
            pct_change = ((bias_aware_val - baseline_val) / abs(baseline_val)) * 100
            direction = "↑" if pct_change > 0 else "↓"
            print(f"  {metric}: {pct_change:+.2f}% {direction}")
    
    print("\n" + "="*60)
    print("Quick test completed!")
    print("\nExpected results (from paper with N=5000):")
    print("  Mean return: -7.2%")
    print("  Volatility: -10.6%")
    print("  VaR improvement: -15.1% (less negative)")
    print("  Stability: +66.2%")
    print("="*60)


def test_individual_biases():
    """
    Test each bias individually (mini ablation study)
    """
    print("\n\n" + "="*60)
    print("Mini Ablation Study - Individual Bias Effects")
    print("="*60)
    
    quick_params = SimulationParameters(
        n_simulations=500,  # Even fewer for ablation
        n_alternatives=30
    )
    
    conditions = {
        'Loss Aversion Only': {
            'bias_params': BiasParameters(lambda_loss_aversion=2.0, 
                                         beta_overconfidence=0.0, 
                                         gamma_herding=0.0),
            'apply_biases': {'loss_aversion': True, 'overconfidence': False, 'herding': False}
        },
        'Overconfidence Only': {
            'bias_params': BiasParameters(lambda_loss_aversion=1.0, 
                                         beta_overconfidence=0.35, 
                                         gamma_herding=0.0),
            'apply_biases': {'loss_aversion': False, 'overconfidence': True, 'herding': False}
        },
        'Herding Only': {
            'bias_params': BiasParameters(lambda_loss_aversion=1.0, 
                                         beta_overconfidence=0.0, 
                                         gamma_herding=0.25),
            'apply_biases': {'loss_aversion': False, 'overconfidence': False, 'herding': True}
        }
    }
    
    results_summary = []
    
    for condition_name, config in conditions.items():
        print(f"\nTesting: {condition_name}")
        
        sim = MonteCarloSimulation(
            quick_params,
            config['bias_params'],
            config['apply_biases']
        )
        
        results = sim.run_full_simulation()
        metrics = sim.compute_aggregate_metrics(results)
        
        results_summary.append({
            'Condition': condition_name,
            'Mean Return': metrics['mean_return'],
            'Volatility': metrics['return_volatility'],
            'VaR (5%)': metrics['var_5pct'],
            'Stability': metrics['decision_stability']
        })
    
    df = pd.DataFrame(results_summary)
    print("\n\nSummary Table:")
    print(df.to_string(index=False))
    
    print("\n\nExpected patterns from paper:")
    print("  Loss Aversion: Reduces VaR (~8.3%), increases stability")
    print("  Overconfidence: Increases volatility (~12.1%), decreases stability")
    print("  Herding: Modest VaR reduction (~2.6%), moderate stability increase")


def demonstrate_bias_mechanisms():
    """
    Demonstrate how each bias transformation works
    """
    print("\n\n" + "="*60)
    print("Bias Mechanism Demonstration")
    print("="*60)
    
    from behavioral_bias_simulation import BehavioralBiasModel
    
    bias_model = BehavioralBiasModel(
        BiasParameters(
            lambda_loss_aversion=2.0,
            beta_overconfidence=0.35,
            gamma_herding=0.25
        )
    )
    
    print("\n1. Loss Aversion Transformation")
    print("-" * 40)
    test_returns = [-0.10, -0.05, 0.0, 0.05, 0.10]
    print("Reference point: 0.0")
    print("Loss aversion coefficient (λ): 2.0")
    print("\nReturn → Perceived Value:")
    for r in test_returns:
        perceived = bias_model.apply_loss_aversion(r)
        print(f"  {r:+.2f} → {perceived:+.2f}")
    
    print("\n2. Overconfidence Transformation")
    print("-" * 40)
    test_volatilities = [0.10, 0.15, 0.20, 0.25, 0.30]
    print("Overconfidence intensity (β): 0.35")
    print("\nTrue Volatility → Perceived Volatility:")
    for vol in test_volatilities:
        perceived = bias_model.apply_overconfidence(vol)
        reduction = ((vol - perceived) / vol) * 100
        print(f"  {vol:.2f} → {perceived:.2f} ({reduction:.1f}% underestimation)")
    
    print("\n3. Herding Transformation")
    print("-" * 40)
    print("Herding intensity (γ): 0.25")
    print("\nExample:")
    private_score = 0.80
    consensus_score = 0.60
    adjusted = bias_model.apply_herding(private_score, consensus_score)
    print(f"  Private score: {private_score:.2f}")
    print(f"  Consensus score: {consensus_score:.2f}")
    print(f"  Adjusted score: {adjusted:.2f}")
    print(f"  (25% weight on consensus, 75% on private)")


if __name__ == "__main__":
    # Run all tests
    quick_test()
    test_individual_biases()
    demonstrate_bias_mechanisms()
    
    print("\n\n" + "="*60)
    print("All quick tests completed!")
    print("\nTo run the full simulation (N=5000), execute:")
    print("  python behavioral_bias_simulation.py")
    print("="*60)
