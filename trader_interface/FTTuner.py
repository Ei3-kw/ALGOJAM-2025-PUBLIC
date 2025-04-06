import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Load the FT price data
data = pd.read_csv('data/Fintech Token_price_history.csv')

# Create a class to simulate different trading strategies
class FTStrategy:
    def __init__(self, prices, ema_period=10, threshold=0.01, lookback=5):
        self.prices = prices
        self.ema_period = ema_period
        self.threshold = threshold
        self.lookback = lookback
        self.position = 0
        self.position_limit = 35  # From your position limit
        self.cash = 100000  # Starting cash
        self.portfolio_values = [self.cash]
        self.trades = []
        
    def calculate_ema(self, period):
        """Calculate EMA for the given period"""
        if len(self.prices) < period:
            return None
        
        # Calculate EMA
        multiplier = 2 / (period + 1)
        ema = [self.prices[0]]  # Start with first price
        
        for i in range(1, len(self.prices)):
            ema.append((self.prices[i] * multiplier) + (ema[i-1] * (1 - multiplier)))
        
        return ema
    
    def calculate_volatility(self, window):
        """Calculate rolling volatility"""
        if len(self.prices) < window:
            return [0] * len(self.prices)
            
        volatility = [0] * (window-1)
        for i in range(window-1, len(self.prices)):
            price_window = self.prices[i-window+1:i+1]
            volatility.append(np.std(price_window))
            
        return volatility
    
    def calculate_momentum(self, window):
        """Calculate price momentum"""
        if len(self.prices) < window:
            return [0] * len(self.prices)
            
        momentum = [0] * window
        for i in range(window, len(self.prices)):
            momentum.append(self.prices[i] - self.prices[i-window])
            
        return momentum
    
    def backtest(self):
        """Run backtest with strategy parameters"""
        ema = self.calculate_ema(self.ema_period)
        volatility = self.calculate_volatility(self.lookback)
        momentum = self.calculate_momentum(self.lookback)
        
        if ema is None:
            return 0  # Not enough data
        
        cash = self.cash
        position = 0
        portfolio_values = [cash]
        trades = []
        
        # Skip initial days where we don't have indicators
        start_day = max(self.ema_period, self.lookback)
        
        for i in range(start_day, len(self.prices)):
            price = self.prices[i]
            
            # Check for dual signal (price vs EMA and momentum)
            price_signal = price < ema[i] if ema[i] else 0  # True = bullish (price below EMA)
            mom_signal = momentum[i] > 0  # True = bullish (positive momentum)
            volatility_factor = volatility[i] / price if price > 0 else 0
            
            # Trading logic with adaptive position sizing based on volatility
            target_position = 0
            
            # High vol + price below EMA + positive momentum = Strong buy
            if price_signal and mom_signal and volatility_factor > self.threshold:
                target_position = self.position_limit
            # Low vol + price below EMA = Medium buy
            elif price_signal and volatility_factor <= self.threshold:
                target_position = int(self.position_limit * 0.5)
            # High vol + price above EMA + negative momentum = Strong sell
            elif not price_signal and not mom_signal and volatility_factor > self.threshold:
                target_position = -self.position_limit
            # Low vol + price above EMA = Medium sell
            elif not price_signal and volatility_factor <= self.threshold:
                target_position = int(-self.position_limit * 0.5)
            
            # Execute trade if position changes
            if target_position != position:
                trade_size = target_position - position
                trade_value = trade_size * price
                
                # Record trade
                if trade_size != 0:
                    trades.append({
                        'day': i,
                        'price': price,
                        'size': trade_size,
                        'value': trade_value
                    })
                
                # Update position and cash
                position = target_position
                cash -= trade_value
            
            # Calculate portfolio value
            portfolio_value = cash + (position * price)
            portfolio_values.append(portfolio_value)
        
        # Calculate final performance
        self.final_value = portfolio_values[-1]
        self.return_pct = (self.final_value / self.cash - 1) * 100
        self.trades = trades
        self.portfolio_values = portfolio_values
        
        return self.return_pct
    
    def plot_performance(self):
        """Plot strategy performance"""
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_values)
        plt.title(f'Portfolio Value (Return: {self.return_pct:.2f}%)')
        plt.grid(True)
        
        # Plot price and trades
        plt.subplot(2, 1, 2)
        plt.plot(self.prices)
        
        # Plot buy and sell points
        for trade in self.trades:
            if trade['size'] > 0:
                plt.scatter(trade['day'], trade['price'], color='green', marker='^')
            else:
                plt.scatter(trade['day'], trade['price'], color='red', marker='v')
        
        plt.title('FT Price with Trade Signals')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Function to run parameter grid search
def grid_search():
    prices = data['Price'].values
    
    # Parameters to test
    ema_periods = [5, 8, 10, 15, 20, 25]
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.025]
    lookbacks = [3, 5, 7, 10, 15]
    
    results = []
    
    # Test all combinations
    for ema_period, threshold, lookback in product(ema_periods, thresholds, lookbacks):
        strategy = FTStrategy(prices, ema_period, threshold, lookback)
        return_pct = strategy.backtest()
        
        results.append({
            'ema_period': ema_period,
            'threshold': threshold,
            'lookback': lookback,
            'return_pct': return_pct
        })
    
    # Convert to DataFrame and sort by return
    results_df = pd.DataFrame(results)
    best_params = results_df.sort_values('return_pct', ascending=False).head(10)
    
    return best_params

# Run parameter optimization
best_params = grid_search()
print("Top 10 Parameter Combinations:")
print(best_params)

# Visualize best strategy
best_row = best_params.iloc[0]
best_strategy = FTStrategy(
    data['Price'].values,
    ema_period=int(best_row['ema_period']),
    threshold=best_row['threshold'],
    lookback=int(best_row['lookback'])
)
best_strategy.backtest()
best_strategy.plot_performance()

# Function to test a combined strategy that adapts to market regime
def test_adaptive_strategy():
    prices = data['Price'].values
    
    # Stable market parameters (low volatility)
    stable_ema = 20
    stable_threshold = 0.01
    stable_lookback = 10
    
    # Volatile market parameters (high volatility)
    volatile_ema = 8
    volatile_threshold = 0.02
    volatile_lookback = 5
    
    # Detect market regime using 50-day volatility
    vol_window = 50
    volatility = np.zeros(len(prices))
    for i in range(vol_window, len(prices)):
        volatility[i] = np.std(prices[i-vol_window:i])
    
    # Normalize volatility
    mean_vol = np.mean(volatility[vol_window:])
    std_vol = np.std(volatility[vol_window:])
    norm_vol = (volatility - mean_vol) / std_vol
    
    # Implement adaptive strategy
    cash = 100000
    position = 0
    portfolio_values = [cash]
    trades = []
    
    for i in range(vol_window, len(prices)):
        # Select strategy based on market regime
        if norm_vol[i] > 0.5:  # High volatility regime
            ema_period = volatile_ema
            threshold = volatile_threshold
            lookback = volatile_lookback
        else:  # Low volatility regime
            ema_period = stable_ema
            threshold = stable_threshold
            lookback = stable_lookback
        
        # Calculate indicators for current parameters
        if i < ema_period:
            continue
            
        # Simple EMA calculation
        alpha = 2 / (ema_period + 1)
        ema = prices[i-ema_period]
        for j in range(i-ema_period+1, i+1):
            ema = alpha * prices[j] + (1 - alpha) * ema
        
        # Calculate momentum
        momentum = prices[i] - prices[i-lookback]
        
        # Calculate local volatility
        local_vol = np.std(prices[i-lookback:i+1]) / prices[i]
        
        # Trading rules
        price = prices[i]
        price_signal = price < ema  # True = bullish (price below EMA)
        mom_signal = momentum > 0   # True = bullish (positive momentum)
        
        # Determine position
        target_position = 0
        position_limit = 35
        
        if price_signal and mom_signal and local_vol > threshold:
            target_position = position_limit
        elif price_signal and local_vol <= threshold:
            target_position = int(position_limit * 0.5)
        elif not price_signal and not mom_signal and local_vol > threshold:
            target_position = -position_limit
        elif not price_signal and local_vol <= threshold:
            target_position = int(-position_limit * 0.5)
        
        # Execute trade if position changes
        if target_position != position:
            trade_size = target_position - position
            trade_value = trade_size * price
            
            # Record trade
            if trade_size != 0:
                trades.append({
                    'day': i,
                    'price': price,
                    'size': trade_size,
                    'value': trade_value,
                    'regime': 'volatile' if norm_vol[i] > 0.5 else 'stable'
                })
            
            # Update position and cash
            position = target_position
            cash -= trade_value
        
        # Calculate portfolio value
        portfolio_value = cash + (position * price)
        portfolio_values.append(portfolio_value)
    
    # Calculate performance
    final_value = portfolio_values[-1]
    return_pct = (final_value / 100000 - 1) * 100
    
    print(f"Adaptive Strategy Return: {return_pct:.2f}%")
    
    # Plot performance
    plt.figure(figsize=(15, 12))
    
    # Plot portfolio value
    plt.subplot(3, 1, 1)
    plt.plot(portfolio_values)
    plt.title(f'Adaptive Strategy Portfolio Value (Return: {return_pct:.2f}%)')
    plt.grid(True)
    
    # Plot price and trades
    plt.subplot(3, 1, 2)
    plt.plot(prices)
    
    # Plot buy and sell points with regime color
    for trade in trades:
        if trade['size'] > 0:
            color = 'darkgreen' if trade['regime'] == 'volatile' else 'lightgreen'
            plt.scatter(trade['day'], trade['price'], color=color, marker='^')
        else:
            color = 'darkred' if trade['regime'] == 'volatile' else 'lightcoral'
            plt.scatter(trade['day'], trade['price'], color=color, marker='v')
    
    plt.title('FT Price with Trade Signals (Green=Buy, Red=Sell, Dark=Volatile, Light=Stable)')
    plt.grid(True)
    
    # Plot market regime
    plt.subplot(3, 1, 3)
    plt.plot(norm_vol)
    plt.axhline(0.5, color='red', linestyle='--')
    plt.title('Normalized Market Volatility (Above red line = Volatile Regime)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return return_pct, trades, portfolio_values

# Run adaptive strategy test
print("\nTesting Adaptive Strategy:")
adaptive_return, adaptive_trades, adaptive_values = test_adaptive_strategy()