# performance.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Algorithm
from simulation import TradingEngine, quantize_decimal
from decimal import Decimal, ROUND_HALF_UP

class PerformanceAnalyzer:
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
        self.total_return_history = trading_engine.totalReturnHistory
        self.instrument_returns = trading_engine.returnsHistory

    def calculate_metrics(self, risk_free_rate=0.02/252):  # Default annualized risk-free rate of 2% converted to daily
        """Calculate performance metrics including standard deviation, avg daily return, and Sharpe ratio"""

        # Convert to numpy array for calculations
        returns = np.array(self.total_return_history)

        # Calculate metrics for overall strategy
        self.metrics = {
            'total_days': len(returns),
            'total_pnl': self.trading_engine.totalPNL,
            'avg_daily_return': np.mean(returns),
            'std_dev': np.std(returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0,
            'profit_factor': np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.sum(returns[returns < 0]) != 0 else float('inf')
        }

        # Calculate Sharpe Ratio
        if self.metrics['std_dev'] > 0:
            self.metrics['sharpe_ratio'] = (float(self.metrics['avg_daily_return']) - risk_free_rate) / float(self.metrics['std_dev'])
            # Annualized Sharpe Ratio (assuming 252 trading days)
            self.metrics['annualized_sharpe'] = self.metrics['sharpe_ratio'] * np.sqrt(252)
        else:
            self.metrics['sharpe_ratio'] = float('nan')
            self.metrics['annualized_sharpe'] = float('nan')

        # Calculate per-instrument metrics
        self.instrument_metrics = {}
        for instrument, instrument_returns in self.instrument_returns.items():
            inst_returns = np.array(instrument_returns)

            if len(inst_returns) > 0:
                self.instrument_metrics[instrument] = {
                    'total_return': sum(inst_returns),
                    'avg_daily_return': np.mean(inst_returns),
                    'std_dev': np.std(inst_returns),
                    'win_rate': np.sum(inst_returns > 0) / len(inst_returns),
                    'profit_factor': np.sum(inst_returns[inst_returns > 0]) / abs(np.sum(inst_returns[inst_returns < 0])) if np.sum(inst_returns[inst_returns < 0]) != 0 else float('inf')
                }

                # Calculate instrument Sharpe Ratio
                if self.instrument_metrics[instrument]['std_dev'] > 0:
                    self.instrument_metrics[instrument]['sharpe_ratio'] = (
                        float(self.instrument_metrics[instrument]['avg_daily_return']) - risk_free_rate
                    ) / float(self.instrument_metrics[instrument]['std_dev'])
                    # Annualized Sharpe
                    self.instrument_metrics[instrument]['annualized_sharpe'] = (
                        self.instrument_metrics[instrument]['sharpe_ratio'] * np.sqrt(252)
                    )
                else:
                    self.instrument_metrics[instrument]['sharpe_ratio'] = float('nan')
                    self.instrument_metrics[instrument]['annualized_sharpe'] = float('nan')

    def _calculate_max_drawdown(self):
        """Calculate the maximum drawdown from peak to trough"""
        # Calculate cumulative returns
        cumulative = np.cumsum(self.total_return_history)

        # Find the maximum drawdown
        peak = np.maximum.accumulate(cumulative)

        # Handle division by zero - create a safe version of peak for division
        safe_peak = np.where(peak > 0, peak, 1)  # Replace zeros with 1.0 to avoid division by zero
        drawdown = np.where(peak > 0, (peak - cumulative) / safe_peak, 0.0)

        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        return max_drawdown

    def print_performance_report(self):
        """Print a detailed performance report"""
        print("\n" + "="*80)
        print(" "*30 + "PERFORMANCE REPORT")
        print("="*80)

        print(f"\nOVERALL STRATEGY PERFORMANCE:")
        print(f"Total Trading Days: {self.metrics['total_days']}")
        print(f"Total P&L: ${self.metrics['total_pnl']:.2f}")
        print(f"Average Daily Return: ${self.metrics['avg_daily_return']:.2f}")
        print(f"Standard Deviation: ${self.metrics['std_dev']:.2f}")
        print(f"Sharpe Ratio (Daily): {self.metrics['sharpe_ratio']:.4f}")
        print(f"Sharpe Ratio (Annualized): {self.metrics['annualized_sharpe']:.4f}")
        print(f"Maximum Drawdown: {self.metrics['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {self.metrics['win_rate']*100:.2f}%")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")

        print("\n" + "-"*80)
        print("INSTRUMENT PERFORMANCE:")

        # Create a sorted list of instruments by total return
        sorted_instruments = sorted(
            self.instrument_metrics.items(),
            key=lambda x: x[1]['total_return'],
            reverse=True
        )

        # Print header
        header = f"{'Instrument':<25} {'Total Return':<15} {'Avg Daily':<15} {'Std Dev':<15} {'Sharpe':<10} {'Win Rate':<10} {'Profit Factor':<15}"
        print(header)
        print("-" * 105)

        # Print instrument metrics
        for instrument, metrics in sorted_instruments:
            instrument_name = instrument[:22] + "..." if len(instrument) > 25 else instrument
            row = (
                f"{instrument_name:<25} "
                f"${metrics['total_return']:<14.2f} "
                f"${metrics['avg_daily_return']:<14.2f} "
                f"${metrics['std_dev']:<14.2f} "
                f"{metrics['annualized_sharpe']:<10.2f} "
                f"{metrics['win_rate']*100:<9.1f}% "
                f"{metrics['profit_factor']:<15.2f}"
            )
            print(row)

        print("="*80)

    def plot_metrics(self):
        """Create additional performance visualizations"""
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Daily Returns Distribution
        returns = np.array(self.total_return_history)
        axs[0, 0].hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axs[0, 0].axvline(x=0, color='r', linestyle='--')
        axs[0, 0].axvline(x=np.mean(returns), color='g', linestyle='-')
        axs[0, 0].set_title('Daily Returns Distribution')
        axs[0, 0].set_xlabel('Daily Return ($)')
        axs[0, 0].set_ylabel('Frequency')

        # Plot 2: Cumulative Returns
        cumulative = np.cumsum(returns)
        axs[0, 1].plot(cumulative, color='blue')
        axs[0, 1].set_title('Cumulative Returns')
        axs[0, 1].set_xlabel('Trading Day')
        axs[0, 1].set_ylabel('Cumulative P&L ($)')

        # Plot 3: Drawdown
        peak = np.maximum.accumulate(cumulative)
        # Safe division for drawdown calculation
        safe_peak = np.where(peak > 0, peak, 1.0)
        drawdown = np.where(peak > 0, (peak - cumulative) / safe_peak, 0.0)

        axs[1, 0].fill_between(range(len(drawdown)), 0, -drawdown*100, color='red', alpha=0.3)
        axs[1, 0].set_title('Drawdown')
        axs[1, 0].set_xlabel('Trading Day')
        axs[1, 0].set_ylabel('Drawdown (%)')
        axs[1, 0].set_ylim(bottom=-max(drawdown)*100*1.1 if max(drawdown) > 0 else -5)

        # Plot 4: Instrument Contribution
        instruments = list(self.instrument_metrics.keys())
        total_returns = [self.instrument_metrics[inst]['total_return'] for inst in instruments]

        # Use a colormap with enough colors for all instruments
        colors = plt.cm.viridis(np.linspace(0, 1, len(instruments)))

        # For better readability with many instruments, use shortened names
        short_names = [inst[:10] + '...' if len(inst) > 10 else inst for inst in instruments]

        bars = axs[1, 1].bar(short_names, total_returns, color=colors)
        axs[1, 1].set_title('Instrument Contribution to Total P&L')
        axs[1, 1].set_xlabel('Instrument')
        axs[1, 1].set_ylabel('Total Return ($)')
        axs[1, 1].set_xticklabels(short_names, rotation=45, ha='right')

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            label_position = height + (5 if height >= 0 else -15)
            axs[1, 1].text(
                bar.get_x() + bar.get_width()/2.,
                label_position,
                f'${height:.0f}',  # Rounded to nearest dollar for readability
                ha='center',
                va='bottom' if height >= 0 else 'top',
                rotation=45,
                fontsize=8
            )

        plt.tight_layout()

        # Save the figure
        output_dir = './simulation_results'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300)
        plt.show()

    def save_metrics_to_csv(self):
        """Save performance metrics to CSV files"""
        output_dir = './simulation_results'
        os.makedirs(output_dir, exist_ok=True)

        # Save overall metrics
        overall_df = pd.DataFrame({k: [v] for k, v in self.metrics.items()})
        overall_df.to_csv(os.path.join(output_dir, 'overall_metrics.csv'), index=False)

        # Save instrument metrics
        instruments = []
        metrics_data = {}

        for instrument, metrics in self.instrument_metrics.items():
            instruments.append(instrument)
            for metric_name, metric_value in metrics.items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(metric_value)

        instrument_df = pd.DataFrame({'instrument': instruments, **metrics_data})
        instrument_df.to_csv(os.path.join(output_dir, 'instrument_metrics.csv'), index=False)

        print(f"Metrics saved to {output_dir}")

def run_with_metrics():
    """Run the simulation and calculate performance metrics"""
    print("Starting trading simulation with performance metrics...")

    # Run the trading engine
    engine = TradingEngine()
    algorithm_instance = Algorithm(engine.positions)
    engine.run_algorithms(algorithm_instance)

    # Calculate and display performance metrics
    analyzer = PerformanceAnalyzer(engine)
    analyzer.calculate_metrics()
    analyzer.print_performance_report()
    analyzer.save_metrics_to_csv()
    
    # Generate plots
    engine.plot_returns()  # Original plots
    analyzer.plot_metrics()  # Additional performance plots
    
    print("Simulation and performance analysis complete.")
    return engine, analyzer

if __name__ == "__main__":
    engine, analyzer = run_with_metrics()