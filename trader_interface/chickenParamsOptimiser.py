# Parameter Optimization Script for Trading Algorithm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Algorithm
from simulation import TradingEngine

class OptimizedAlgorithm(Algorithm):
    """Extended Algorithm class to allow parameter customization"""

    def __init__(self, positions, lookback, z_threshold_short, z_threshold_long):
        super().__init__(positions)
        self.lookback = lookback
        self.z_threshold_short = z_threshold_short
        self.z_threshold_long = z_threshold_long

def run_backtest(lookback, z_threshold_short, z_threshold_long):
    """Run a backtest with specific parameters"""
    try:
        engine = TradingEngine()  # Removed the verbose parameter
        algorithmInstance = OptimizedAlgorithm(
            engine.positions,
            lookback=lookback,
            z_threshold_short=z_threshold_short,
            z_threshold_long=z_threshold_long
        )

        # Suppress print statements from the engine
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Run the algorithm
        engine.run_algorithms(algorithmInstance)

        # Restore stdout
        sys.stdout = old_stdout

        return engine.totalPNL, algorithmInstance.performance
    except Exception as e:
        print(f"Error during backtest: {e}")
        return None, None

def parameter_grid_search():
    """Test various parameter combinations and find the optimal ones"""
    # Define parameter ranges to test - reduced for faster testing
    lookback_range = [4, 5, 6, 7]
    z_threshold_short_range = np.arange(0.01, 0.09, 0.005)
    z_threshold_long_range = np.arange(0.01, 0.02, 0.002)

    # Store results
    results = []

    # Track best parameters
    best_pnl = float('-inf')
    best_params = None

    total_combinations = len(lookback_range) * len(z_threshold_short_range) * len(z_threshold_long_range)
    completed = 0

    print(f"Starting grid search with {total_combinations} parameter combinations...")

    # First test the default parameters
    default_params = {
        'lookback': 5,
        'z_threshold_short': 0.02,
        'z_threshold_long': 0.02
    }

    print("Testing default parameters first...")
    pnl, performance = run_backtest(
        default_params['lookback'],
        default_params['z_threshold_short'],
        default_params['z_threshold_long']
    )

    if pnl is not None:
        signals = performance.get('signal', [])
        signal_changes = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1]) if signals else 0

        result = {
            'lookback': default_params['lookback'],
            'z_threshold_short': default_params['z_threshold_short'],
            'z_threshold_long': default_params['z_threshold_long'],
            'pnl': pnl,
            'signal_changes': signal_changes
        }
        results.append(result)
        best_pnl = pnl
        best_params = result.copy()
        print(f"Default parameters PNL: {pnl}")
    else:
        print("Failed to run backtest with default parameters.")
        return None, None

    # Now test the grid
    try:
        # Loop through all combinations
        for lookback in lookback_range:
            for z_short in z_threshold_short_range:
                for z_long in z_threshold_long_range:
                    # Skip testing the default parameters again
                    if (lookback == default_params['lookback'] and
                        round(z_short, 2) == default_params['z_threshold_short'] and
                        round(z_long, 2) == default_params['z_threshold_long']):
                        continue

                    completed += 1
                    print(f"Testing combination {completed}/{total_combinations}: "
                          f"lookback={lookback}, z_short={z_short:.2f}, z_long={z_long:.2f}")

                    pnl, performance = run_backtest(lookback, z_short, z_long)

                    if pnl is not None:
                        # Calculate additional performance metrics
                        signals = performance.get('signal', [])
                        if signals:
                            signal_changes = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1])
                        else:
                            signal_changes = 0

                        # Record results
                        result = {
                            'lookback': lookback,
                            'z_threshold_short': round(z_short, 2),
                            'z_threshold_long': round(z_long, 2),
                            'pnl': pnl,
                            'signal_changes': signal_changes
                        }
                        results.append(result)

                        # Update best parameters if current PNL is better
                        if pnl > best_pnl:
                            best_pnl = pnl
                            best_params = result.copy()
                            print(f"New best parameters found: lookback={lookback}, "
                                  f"z_short={z_short:.2f}, z_long={z_long:.2f}, PNL={pnl}")

                    # Option to save intermediate results periodically
                    if completed % 10 == 0 and results:
                        temp_df = pd.DataFrame(results)
                        temp_df.to_csv('parameter_optimization_interim_results.csv', index=False)

    except KeyboardInterrupt:
        print("\nSearch interrupted by user. Saving current results...")

    # Create DataFrame with results
    if results:
        results_df = pd.DataFrame(results)

        # Save results to CSV
        results_df.to_csv('parameter_optimization_results.csv', index=False)

        # Return the best parameters and results DataFrame
        return best_params, results_df
    else:
        print("No valid results found. Please check your implementation.")
        return None, None

def visualize_results(results_df):
    """Create visualizations to help understand parameter impact"""
    if results_df is None or len(results_df) == 0:
        print("No results to visualize.")
        return

    # Create output directory if it doesn't exist
    os.makedirs('./optimization_results', exist_ok=True)

    # 1. Plot PNL distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['pnl'], bins=min(30, len(results_df)), alpha=0.7)
    plt.axvline(results_df['pnl'].max(), color='r', linestyle='dashed',
                label=f'Max PNL: {results_df["pnl"].max():.2f}')
    plt.title('Distribution of PNL Across Parameter Combinations')
    plt.xlabel('PNL')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('./optimization_results/pnl_distribution.png')

    # 2. Top performing parameter combinations for each lookback
    if len(results_df) > 1:
        best_by_lookback = results_df.sort_values('pnl', ascending=False).groupby('lookback').head(1)

        plt.figure(figsize=(12, 8))
        plt.scatter(best_by_lookback['lookback'], best_by_lookback['pnl'])
        for _, row in best_by_lookback.iterrows():
            plt.annotate(f"short={row['z_threshold_short']}, long={row['z_threshold_long']}",
                         (row['lookback'], row['pnl']),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')
        plt.title('Best PNL by Lookback Period')
        plt.xlabel('Lookback Period')
        plt.ylabel('PNL')
        plt.grid(True)
        plt.savefig('./optimization_results/best_pnl_by_lookback.png')

    # 3. Plot relationship between parameters and performance
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Lookback vs PNL
    if len(results_df['lookback'].unique()) > 1:
        lookback_avg = results_df.groupby('lookback')['pnl'].mean().reset_index()
        axs[0].plot(lookback_avg['lookback'], lookback_avg['pnl'], marker='o')
        axs[0].set_title('Average PNL by Lookback Period')
        axs[0].set_xlabel('Lookback Period')
        axs[0].set_ylabel('Average PNL')
        axs[0].grid(True)
    else:
        axs[0].text(0.5, 0.5, 'Insufficient data for lookback analysis',
                   horizontalalignment='center', verticalalignment='center')

    # z_threshold_short vs PNL
    if len(results_df['z_threshold_short'].unique()) > 1:
        z_short_avg = results_df.groupby('z_threshold_short')['pnl'].mean().reset_index()
        axs[1].plot(z_short_avg['z_threshold_short'], z_short_avg['pnl'], marker='o')
        axs[1].set_title('Average PNL by Z-Threshold Short')
        axs[1].set_xlabel('Z-Threshold Short')
        axs[1].set_ylabel('Average PNL')
        axs[1].grid(True)
    else:
        axs[1].text(0.5, 0.5, 'Insufficient data for z_threshold_short analysis',
                   horizontalalignment='center', verticalalignment='center')

    # z_threshold_long vs PNL
    if len(results_df['z_threshold_long'].unique()) > 1:
        z_long_avg = results_df.groupby('z_threshold_long')['pnl'].mean().reset_index()
        axs[2].plot(z_long_avg['z_threshold_long'], z_long_avg['pnl'], marker='o')
        axs[2].set_title('Average PNL by Z-Threshold Long')
        axs[2].set_xlabel('Z-Threshold Long')
        axs[2].set_ylabel('Average PNL')
        axs[2].grid(True)
    else:
        axs[2].text(0.5, 0.5, 'Insufficient data for z_threshold_long analysis',
                   horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.savefig('./optimization_results/parameter_impact.png')

    # Create a 3D plot to visualize the relationship between z-thresholds
    if (len(results_df['z_threshold_short'].unique()) > 1 and
        len(results_df['z_threshold_long'].unique()) > 1):

        # For each lookback, create a heatmap of short vs long thresholds
        for lookback in results_df['lookback'].unique():
            subset = results_df[results_df['lookback'] == lookback]

            if len(subset) > 1:
                pivot_table = subset.pivot_table(
                    index='z_threshold_short',
                    columns='z_threshold_long',
                    values='pnl'
                )

                plt.figure(figsize=(10, 8))
                heatmap = plt.imshow(pivot_table, cmap='viridis',
                                    interpolation='nearest', aspect='auto')
                plt.colorbar(heatmap, label='PNL')
                plt.title(f'PNL Heatmap - Lookback {lookback}')
                plt.xlabel('Z-Threshold Long Index')
                plt.ylabel('Z-Threshold Short Index')

                # Add x and y tick labels
                plt.xticks(range(len(pivot_table.columns)),
                          [f"{x:.2f}" for x in pivot_table.columns],
                          rotation=90)
                plt.yticks(range(len(pivot_table.index)),
                          [f"{y:.2f}" for y in pivot_table.index])

                plt.tight_layout()
                plt.savefig(f'./optimization_results/heatmap_lookback_{lookback}.png')

    # Close all figures
    plt.close('all')

def optimize_parameters():
    print("Starting parameter optimization...")

    # Run grid search
    best_params, results_df = parameter_grid_search()

    # Display best parameters if found
    if best_params:
        print("\n" + "=" * 50)
        print("OPTIMIZATION COMPLETE")
        print("=" * 50)
        print(f"Best parameters found:")
        print(f"  lookback = {best_params['lookback']}")
        print(f"  z_threshold_short = {best_params['z_threshold_short']}")
        print(f"  z_threshold_long = {best_params['z_threshold_long']}")
        print(f"  Resulting PNL: {best_params['pnl']}")
        print("=" * 50)

        # Top parameter combinations
        if results_df is not None and len(results_df) >= 5:
            print("\nTop 5 parameter combinations:")
            top_n = results_df.sort_values('pnl', ascending=False).head(5)
            print(top_n[['lookback', 'z_threshold_short', 'z_threshold_long', 'pnl']])

        # Generate visualizations
        print("\nGenerating visualizations...")
        visualize_results(results_df)

        if results_df is not None:
            print("\nResults saved to 'parameter_optimization_results.csv'")
            print("Visualizations saved to 'optimization_results' directory")
    else:
        print("\nNo optimal parameters found. Please check your implementation or adjust search ranges.")

if __name__ == "__main__":
    optimize_parameters()