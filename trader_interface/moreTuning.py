# Test script for asymmetric z-thresholds in different spread relationships
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Algorithm
from simulation import TradingEngine  # Import the trading engine

class EnhancedAlgorithm(Algorithm):
    """Extended Algorithm class with asymmetric Z-thresholds for different spreads"""
    
    def __init__(self, positions, lookback=5, 
                 z_threshold_short_rc=0.02, z_threshold_long_rc=0.02,
                 z_threshold_short_ss=0.02, z_threshold_long_ss=0.02):
        super().__init__(positions)
        self.lookback = lookback
        # RC-FC spread thresholds
        self.z_threshold_short_rc = z_threshold_short_rc
        self.z_threshold_long_rc = z_threshold_long_rc
        # SS-FC spread thresholds
        self.z_threshold_short_ss = z_threshold_short_ss
        self.z_threshold_long_ss = z_threshold_long_ss
        
        # Keep these at default values rather than None
        self.z_threshold_short = 0.02
        self.z_threshold_long = 0.02

        # Add tracking for individual spread performances
        self.performance_rc = {'day': [], 'spread': [], 'z_score': [], 'signal': []}
        self.performance_ss = {'day': [], 'spread': [], 'z_score': [], 'signal': []}

    # Override calculate_z_score to handle both integer lookback and dict performance
    def calculate_z_score(self, actual, fair_value, lookback_or_performance=None):
        """
        Calculate z-score with flexible parameter handling
        - If lookback_or_performance is an integer: use it as lookback period
        - If lookback_or_performance is a dict: use it as performance data
        - If lookback_or_performance is None: use self.lookback and self.performance
        """
        # Determine if third parameter is lookback (int) or performance data (dict)
        performance_data = self.performance  # Default
        lookback_period = self.lookback      # Default

        if lookback_or_performance is not None:
            if isinstance(lookback_or_performance, dict):
                # It's a performance dictionary
                performance_data = lookback_or_performance
            elif isinstance(lookback_or_performance, int):
                # It's a lookback period
                lookback_period = lookback_or_performance

        # Use the historical spread between actual and fair value
        spread = actual - fair_value

        # If not enough history, return 0
        if len(performance_data['spread']) < lookback_period:
            return 0

        # Calculate z-score based on recent spread history
        recent_spreads = performance_data['spread'][-lookback_period:]
        mean_spread = np.mean(recent_spreads)
        std_spread = np.std(recent_spreads)

        # Avoid division by zero
        if std_spread == 0:
            return 0

        return (spread - mean_spread) / std_spread

    # Other overridden methods remain the same
    def calculate_fair_value_rc(self):
        """Calculate fair value for FC based on RC relationship"""
        rc_price = self.get_current_price("Raw Chicken")
        return self.intercept + self.rc_coef * rc_price

    def calculate_fair_value_ss(self):
        """Calculate fair value for FC based on SS relationship"""
        ss_price = self.get_current_price("Secret Spices")
        return self.intercept + self.ss_coef * ss_price
# def run_single_test(lookback, z_short_rc, z_long_rc, z_short_ss, z_long_ss):
#     """Run a single test with the given parameters"""
#     try:
#         engine = TradingEngine()  # Initialize without verbose parameter
#         algo = EnhancedAlgorithm(
#             engine.positions,
#             lookback=lookback,
#             z_threshold_short_rc=z_short_rc,
#             z_threshold_long_rc=z_long_rc,
#             z_threshold_short_ss=z_short_ss,
#             z_threshold_long_ss=z_long_ss
#         )
#         engine.run_algorithms(algo)
#         return engine.totalPNL, algo.performance, algo.performance_rc, algo.performance_ss
#     except Exception as e:
#         print(f"Error during test: {e}")
#         return None, None, None, None

def run_single_test(lookback, z_short_rc, z_long_rc, z_short_ss, z_long_ss):
    """Run a single test with the given parameters"""
    try:
        print(f"Initializing TradingEngine...")
        engine = TradingEngine()  # Initialize without verbose parameter

        print(f"Creating algorithm with params: lookback={lookback}, z_short_rc={z_short_rc}, z_long_rc={z_long_rc}, z_short_ss={z_short_ss}, z_long_ss={z_long_ss}")
        algo = EnhancedAlgorithm(
            engine.positions,
            lookback=lookback,
            z_threshold_short_rc=z_short_rc,
            z_threshold_long_rc=z_long_rc,
            z_threshold_short_ss=z_short_ss,
            z_threshold_long_ss=z_long_ss
        )

        print("Running algorithm...")
        engine.run_algorithms(algo)

        print(f"Type of totalPNL: {type(engine.totalPNL)}, value: {engine.totalPNL}")
        return engine.totalPNL, algo.performance, algo.performance_rc, algo.performance_ss
    except Exception as e:
        import traceback
        print(f"Error during test: {e}")
        print(traceback.format_exc())  # This will print the full stack trace
        return None, None, None, None

def test_parameter_combinations():
    """Test different combinations of parameters"""
    # Define parameter ranges
    lookback_values = range(3, 21)
    
    # RC-FC spread thresholds
    z_short_rc_values = np.arange(0.02, 0.25, 0.1)
    z_long_rc_values = np.arange(0.02, 0.25, 0.1)
    
    # SS-FC spread thresholds
    z_short_ss_values = np.arange(0.02, 0.25, 0.1)
    z_long_ss_values = np.arange(0.02, 0.25, 0.1)
    
    # Store results
    results = []
    
    # First test default values
    print("Testing default parameters...")
    default_params = {
        'lookback': 5,
        'z_short_rc': 0.02,
        'z_long_rc': 0.02,
        'z_short_ss': 0.02,
        'z_long_ss': 0.02
    }
    
    pnl, perf, perf_rc, perf_ss = run_single_test(
        default_params['lookback'],
        default_params['z_short_rc'],
        default_params['z_long_rc'],
        default_params['z_short_ss'],
        default_params['z_long_ss']
    )
    
    if pnl is not None:
        result = {
            'lookback': default_params['lookback'],
            'z_short_rc': default_params['z_short_rc'],
            'z_long_rc': default_params['z_long_rc'],
            'z_short_ss': default_params['z_short_ss'],
            'z_long_ss': default_params['z_long_ss'],
            'pnl': pnl
        }
        results.append(result)
        print(f"Default parameters PNL: {pnl}")
    
    # Test other combinations
    total_combinations = len(lookback_values) * len(z_short_rc_values) * len(z_long_rc_values) * len(z_short_ss_values) * len(z_long_ss_values)
    completed = 0
    
    print(f"Testing {total_combinations} parameter combinations...")
    
    for lookback in lookback_values:
        for z_short_rc in z_short_rc_values:
            for z_long_rc in z_long_rc_values:
                for z_short_ss in z_short_ss_values:
                    for z_long_ss in z_long_ss_values:
                        # Skip if already tested default combination
                        if (lookback == default_params['lookback'] and
                            z_short_rc == default_params['z_short_rc'] and
                            z_long_rc == default_params['z_long_rc'] and
                            z_short_ss == default_params['z_short_ss'] and
                            z_long_ss == default_params['z_long_ss']):
                            continue
                            
                        completed += 1
                        print(f"Testing combination {completed}/{total_combinations-1}...")
                        
                        pnl, perf, perf_rc, perf_ss = run_single_test(
                            lookback, z_short_rc, z_long_rc, z_short_ss, z_long_ss
                        )
                        
                        if pnl is not None:
                            result = {
                                'lookback': lookback,
                                'z_short_rc': z_short_rc,
                                'z_long_rc': z_long_rc,
                                'z_short_ss': z_short_ss,
                                'z_long_ss': z_long_ss,
                                'pnl': pnl
                            }
                            results.append(result)
    
    # Create DataFrame and save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('asymmetric_threshold_results.csv', index=False)
        
        # Show top results
        print("\nTop 5 Parameter Combinations:")
        top_results = results_df.sort_values('pnl', ascending=False).head(5)
        print(top_results)
        
        # Get best result
        best_params = top_results.iloc[0].to_dict()
        print("\n" + "=" * 50)
        print("BEST PARAMETERS:")
        print(f"  lookback = {best_params['lookback']}")
        print(f"  RC-FC spread thresholds:")
        print(f"    z_threshold_short_rc = {best_params['z_short_rc']}")
        print(f"    z_threshold_long_rc = {best_params['z_long_rc']}")
        print(f"  SS-FC spread thresholds:")
        print(f"    z_threshold_short_ss = {best_params['z_short_ss']}")
        print(f"    z_threshold_long_ss = {best_params['z_long_ss']}")
        print(f"  Resulting PNL: {best_params['pnl']}")
        print("=" * 50)
        
        return results_df, best_params
    else:
        print("No valid results found.")
        return None, None

if __name__ == "__main__":
    print("Testing asymmetric z-threshold configurations...")
    results_df, best_params = test_parameter_combinations()
    
    if results_df is not None:
        print("\nResults saved to 'asymmetric_threshold_results.csv'")