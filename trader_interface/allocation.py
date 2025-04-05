import numpy as np
import pandas as pd
from typing import Dict, List

class DynamicAllocationStrategy:
    def __init__(self, total_daily_budget: float = 600000):
        self.total_daily_budget = total_daily_budget
        
        # Define position limits for each instrument
        self.position_limits = {
            "UQ_Dollar": 650,
            "Dawg_Food": 400,
            "Fintech_Token": 35,
            "Fried_Chicken": 30000,
            "Raw_Chicken": 2500,
            "Secret_Spices": 200,
            "Goober_Eats": 75000,
            "QUACK": 40000,
            "Purple_Elixir": 10000,
            "Rare_Watch": 250
        }
        
        # Base allocation percentages (minimum allocation)
        self.base_allocations = {
            "UQ_Dollar": 0.08,
            "Dawg_Food": 0.07,
            "Fintech_Token": 0.10,
            "Fried_Chicken": 0.10,
            "Raw_Chicken": 0.08,
            "Secret_Spices": 0.08,
            "Goober_Eats": 0.09,
            "QUACK": 0.15,
            "Purple_Elixir": 0.10,
            "Rare_Watch": 0.15
        }
        
        # Performance metrics history (will be updated daily)
        self.performance_history = {instrument: [] for instrument in self.position_limits.keys()}
        
        # Volatility estimate for each instrument (updated daily)
        self.volatility_estimates = {instrument: 0.0 for instrument in self.position_limits.keys()}
        
        # Signal strength for each instrument (updated daily)
        self.signal_strength = {instrument: 0.0 for instrument in self.position_limits.keys()}
    
    def update_market_data(self, price_data: Dict[str, pd.DataFrame]):
        """
        Update market data and calculate metrics for each instrument.
        price_data: Dictionary with instrument names as keys and price DataFrames as values
        """
        for instrument, data in price_data.items():
            if len(data) >= 20:  # Ensure we have enough data
                # Calculate recent volatility (20-day)
                self.volatility_estimates[instrument] = data['price'].pct_change().std() * np.sqrt(252)
                
                # Calculate specific signals based on instrument type
                self.signal_strength[instrument] = self._calculate_instrument_signal(instrument, data)
                
                # Update performance history
                if len(data) >= 2:
                    daily_return = (data['price'].iloc[-1] / data['price'].iloc[-2]) - 1
                    self.performance_history[instrument].append(daily_return)
    
    def _calculate_instrument_signal(self, instrument: str, data: pd.DataFrame) -> float:
        """
        Calculate trading signal strength for each instrument based on its characteristics
        """
        if instrument == "UQ_Dollar":
            # Mean reversion signal - deviation from $100 value
            mean_value = 100
            current_price = data['price'].iloc[-1]
            deviation = abs(current_price - mean_value)
            signal = min(deviation / 10, 1.0)  # Normalize signal to [0,1]
            
        elif instrument == "Dawg_Food":
            # Momentum signal
            if len(data) >= 50:
                ma_20 = data['price'].rolling(20).mean().iloc[-1]
                ma_50 = data['price'].rolling(50).mean().iloc[-1]
                signal = (ma_20 / ma_50) - 1  # Positive when short-term trend is stronger
                signal = max(min(signal * 5, 1.0), 0.0)  # Normalize to [0,1]
            else:
                signal = 0.5
                
        elif instrument == "Fintech_Token":
            # Volatility regime detection
            if len(data) >= 30:
                recent_vol = data['price'].pct_change().rolling(10).std().iloc[-1]
                longer_vol = data['price'].pct_change().rolling(30).std().iloc[-1]
                
                # Is it in volatile or stable regime?
                is_volatile = recent_vol > longer_vol * 1.2
                
                # For volatile regime: momentum signal
                # For stable regime: mean reversion signal
                if is_volatile:
                    returns = data['price'].pct_change()
                    momentum = returns.rolling(5).mean().iloc[-1] * 10
                    signal = max(min(abs(momentum), 1.0), 0.0)
                else:
                    # Mean reversion in stable regime
                    ma_20 = data['price'].rolling(20).mean().iloc[-1]
                    deviation = (data['price'].iloc[-1] / ma_20) - 1
                    signal = min(abs(deviation) * 5, 1.0)
            else:
                signal = 0.5
                
        elif instrument == "Fried_Chicken":
            # Look for arbitrage opportunities between Raw Chicken, Secret Spices, and Fried Chicken
            # This is a placeholder - real implementation would compare prices across instruments
            signal = 0.5  # Default middle value
            
        elif instrument in ["Raw_Chicken", "Secret_Spices"]:
            # Look for dips in upward trend
            if len(data) >= 20:
                ma_10 = data['price'].rolling(10).mean().iloc[-1]
                ma_20 = data['price'].rolling(20).mean().iloc[-1]
                
                # Uptrend with recent dip is a good opportunity
                uptrend = ma_10 > ma_20
                recent_dip = data['price'].iloc[-1] < ma_10
                
                if uptrend and recent_dip:
                    dip_magnitude = (ma_10 - data['price'].iloc[-1]) / ma_10
                    signal = min(dip_magnitude * 10, 1.0)
                else:
                    signal = 0.3
            else:
                signal = 0.5
                
        elif instrument == "Goober_Eats":
            # Random fluctuations linked to Fried Chicken
            # This would need data correlation analysis in practice
            signal = np.random.uniform(0.3, 0.7)  # Placeholder
            
        elif instrument == "QUACK":
            # Seasonal pattern detection
            # In real implementation, this would detect where in the seasonal cycle we are
            # and how strong the current seasonal signal is
            if len(data) >= 365:  # If we have a year of data
                # Simple seasonal detection (would be more sophisticated in reality)
                month = pd.Timestamp.now().month
                seasonal_strength = {
                    1: 0.8, 2: 0.7, 3: 0.5, 4: 0.3,
                    5: 0.2, 6: 0.3, 7: 0.5, 8: 0.7,
                    9: 0.9, 10: 1.0, 11: 0.9, 12: 0.8
                }
                signal = seasonal_strength.get(month, 0.5)
            else:
                signal = 0.5
                
        elif instrument == "Purple_Elixir":
            # Complex pattern with seasonal and short-term elements
            # This would use more sophisticated time series modeling in practice
            signal = 0.6  # Placeholder
            
        elif instrument == "Rare_Watch":
            # Looking for outlier detection
            if len(data) >= 20:
                # Simple z-score based outlier detection
                mean_price = data['price'].rolling(20).mean().iloc[-1]
                std_price = data['price'].rolling(20).std().iloc[-1]
                z_score = abs((data['price'].iloc[-1] - mean_price) / std_price)
                
                # High score means potential outlier event
                signal = min(z_score / 3, 1.0)
            else:
                signal = 0.5
        
        else:
            signal = 0.5  # Default for unknown instruments
            
        return signal
    
    def calculate_daily_allocations(self) -> Dict[str, Dict]:
        """
        Calculate daily budget allocations based on signals and performance
        Returns a dictionary with amount and units for each instrument
        """
        # Start with base allocations
        adjusted_allocations = self.base_allocations.copy()
        
        # Adjust based on signal strength and volatility
        adjustment_factor = 0.5  # How much to adjust from base allocation
        
        for instrument in self.position_limits.keys():
            signal = self.signal_strength[instrument]
            vol = self.volatility_estimates[instrument]
            
            # Adjust allocation: Higher signal = more allocation, higher vol = less allocation
            # This is a simplified approach - real models would be more sophisticated
            risk_adjusted_signal = signal / (vol + 0.1)  # Avoid division by zero
            
            # Scale the adjustment
            allocation_adjustment = (risk_adjusted_signal - 0.5) * adjustment_factor
            adjusted_allocations[instrument] += allocation_adjustment
        
        # Ensure allocations are positive and sum to 1
        for instrument in adjusted_allocations:
            adjusted_allocations[instrument] = max(adjusted_allocations[instrument], 0.01)
        
        # Normalize to sum to 1
        total_allocation = sum(adjusted_allocations.values())
        for instrument in adjusted_allocations:
            adjusted_allocations[instrument] /= total_allocation
        
        # Calculate monetary allocation and units
        result = {}
        for instrument, allocation_pct in adjusted_allocations.items():
            budget_amount = allocation_pct * self.total_daily_budget
            
            # Estimate price per unit (in real implementation, this would come from market data)
            # This is a placeholder - would need actual pricing data
            estimated_price = self._get_estimated_price(instrument)
            
            # Calculate units based on budget and price
            max_units = budget_amount / estimated_price
            allocated_units = min(max_units, self.position_limits[instrument])
            actual_budget = allocated_units * estimated_price
            
            result[instrument] = {
                "budget_percentage": allocation_pct,
                "budget_amount": actual_budget,
                "units": allocated_units,
                "signal_strength": self.signal_strength[instrument]
            }
        
        return result
    
    def _get_estimated_price(self, instrument: str) -> float:
        """
        Get estimated price for an instrument. In practice, this would come from market data.
        These are placeholder values for demonstration.
        """
        estimated_prices = {
            "UQ_Dollar": 100,
            "Dawg_Food": 50,
            "Fintech_Token": 2000,
            "Fried_Chicken": 5,
            "Raw_Chicken": 30,
            "Secret_Spices": 200,
            "Goober_Eats": 2,
            "QUACK": 3,
            "Purple_Elixir": 15,
            "Rare_Watch": 5000
        }
        
        return estimated_prices.get(instrument, 100)

if __name__ == "__main__":
    # Initialize strategy
    strategy = DynamicAllocationStrategy()
    
    # In a real implementation, you would update with actual market data
    # This is just a placeholder to show how it would work
    mock_price_data = {
        instrument: pd.DataFrame({
            'price': np.random.normal(100, 10, 100)  # Mock price data
        }) 
        for instrument in strategy.position_limits.keys()
    }
    
    # Update with market data
    strategy.update_market_data(mock_price_data)
    
    # Get daily allocations
    allocations = strategy.calculate_daily_allocations()
    
    # Print allocations
    for instrument, allocation in allocations.items():
        print(f"{instrument}: {allocation['budget_amount']:.2f} AUD, {allocation['units']:.2f} units")

