import numpy as np
import pandas as pd

FT = "Fintech Token"
PE = "Purple Elixir"
QUACK = "Quantum Universal Algorithmic Currency Koin"
DF = "Dawg Food"
UD = "UQ Dollar"
FC = "Fried Chicken"
SS = "Secret Spices"
GE = "Goober Eats"
RC = "Raw Chicken"
RW = "Rare Watch"

# Custom trading Algorithm
class Algorithm():

    ########################################################
    # NO EDITS REQUIRED TO THESE FUNCTIONS
    ########################################################
    # FUNCTION TO SETUP ALGORITHM CLASS
    def __init__(self, positions):
        # Initialise data stores:
        # Historical data of all instruments
        self.data = {}
        # Initialise position limits
        self.positionLimits = {}
        # Initialise the current day as 0
        self.day = 0
        # Initialise the current positions
        self.positions = positions

        self.QUACKick = 0

        self.QUACKStart = None

    # Helper function to fetch the current price of an instrument
    def get_current_price(self, instrument):
        # return most recent price
        return self.data[instrument][-1]
    ########################################################

    # RETURN DESIRED POSITIONS IN DICT FORM
    def get_positions(self):
        # Get current position
        currentPositions = self.positions
        # Get position limits
        positionLimits = self.positionLimits
        
        # Declare a store for desired positions
        desiredPositions = {}
        # Loop through all the instruments you can take positions on.
        for instrument, positionLimit in positionLimits.items():
            # For each instrument initilise desired position to zero
            desiredPositions[instrument] = 0

        # IMPLEMENT CODE HERE TO DECIDE WHAT POSITIONS YOU WANT 
        #######################################################################
        # Display the current trading day
        print("Starting Algorithm for Day:", self.day)
        
        trade_instruments = self.positionLimits.keys() - {RW, FT, QUACK}

        # Display the prices of instruments I want to trade
        for ins in trade_instruments:
            print(f"{ins}: ${self.get_current_price(ins)}")


        # BENCH MARK -- 2039436.45

        # Start trading from Day 2 onwards. Buy if price dropped and sell if price rose compared to the previous day
        if self.day >= 2:
            for ins in trade_instruments:
                # if price has gone down buy
                if self.data[ins][-2] > self.data[ins][-1]:
                    desiredPositions[ins] = positionLimits[ins]
                else:
                    desiredPositions[ins] = -positionLimits[ins]

        # # Define moving average windows
        # short_window = 3  # Short-term MA (5 days)
        # long_window = 20   # Long-term MA (20 days)

        # # Get historical prices for target instrument
        # prices = np.array(self.data[RC])

        # # Calculate moving averages (if we have enough data)
        # if len(prices) >= long_window:
        #     # Calculate short-term moving average
        #     short_ma = np.mean(prices[-short_window:])

        #     # Calculate long-term moving average
        #     long_ma = np.mean(prices[-long_window:])

        #     # Calculate volatility (safely handle return calculation)
        #     if len(prices) >= 6:
        #         price_today = prices[-6:]  # Last 6 days including today
        #         price_yesterday = prices[-7:-1]  # 6 days offset by 1 (yesterday)
        #         daily_returns = (price_today - price_yesterday) / price_yesterday
        #         volatility = np.std(daily_returns)
        #     else:
        #         # If we don't have enough data for volatility calculation
        #         volatility = 0.02  # Default volatility value

        #     # Generate trend signal
        #     if short_ma > long_ma:
        #         signal = 1  # Bullish
        #     else:
        #         signal = -1  # Bearish

        #     # Calculate trend strength (normalized difference between MAs relative to volatility)
        #     ma_diff = (short_ma - long_ma) / prices[-1]  # Normalized difference
        #     trend_strength = abs(ma_diff) / (volatility if volatility > 0 else 0.02)
        #     trend_strength = min(trend_strength, 3)  # Cap trend strength

        #     # Position sizing based on trend strength and position limit
        #     position_limit = positionLimits[RC]
        #     base_position = position_limit * 0.5  # Start with 50% of max position

        #     if signal != 0:
        #         position_size = base_position * (1 + trend_strength * 0.3)  # Adjust by up to 30% based on trend
        #         position_size = signal * min(abs(position_size), position_limit)
        #     else:
        #         position_size = 0

        #     # Set the desired position
        #     desiredPositions[RC] = int(round(position_size))

        # annual_vol_target = 0.25

        # capital = 600000

        # # Calculate daily volatility target
        # daily_vol_target = annual_vol_target * capital / np.sqrt(252)

        # for instrument in trade_instruments:
        #     # Display current price
        #     print(f"{instrument}: ${self.get_current_price(instrument)}")

        #     # Skip if we don't have enough data yet (need at least 120 days)
        #     if len(self.data[instrument]) < 69:
        #         desiredPositions[instrument] = 0
        #         continue

        #     # Get the price history for this instrument
        #     prices = pd.Series(self.data[instrument])

        #     # Calculate fast and slow moving averages
        #     fast_span = 20
        #     slow_span = 120
        #     ewma_fast = prices.ewm(span=fast_span).mean()
        #     ewma_slow = prices.ewm(span=slow_span).mean()

        #     # Calculate the raw forecast
        #     forecast = ewma_fast - ewma_slow

        #     # Normalize the forecast (centered around +/-10)
        #     forecast_normalized = forecast * 10 / forecast.abs().mean() if forecast.abs().mean() > 0 else 0

        #     # Get the most recent forecast value
        #     current_forecast = forecast_normalized.iloc[-1]

        #     # Clip the forecast to a reasonable range (-20 to 20)
        #     current_forecast = max(min(current_forecast, 20), -20)

        #     # Calculate the volatility of returns for position sizing
        #     returns = prices.diff()
        #     volatility = returns.ewm(span=36, min_periods=36).std().iloc[-1]

        #     # Calculate position size based on volatility
        #     if volatility > 0:
        #         # Position = (Normalized forecast / 10) * (Daily volatility target / Volatility of instrument)
        #         position_size = (current_forecast / 10) * (daily_vol_target / volatility)

        #         # Round to integer and respect position limits
        #         position_int = int(position_size)
        #         position_int = max(min(position_int, positionLimits[instrument]), -positionLimits[instrument])

        #         desiredPositions[instrument] = position_int
        #     else:
        #         # If volatility is zero, avoid division by zero
        #         desiredPositions[instrument] = 0

        # # FT
        # if self.day > 2:
        #     if all([self.data[FT][-1] >= self.data[FT][-2],
        #         self.data[FT][-2] >= self.data[FT][-3]]):
        #         desiredPositions[FT] = -positionLimits[FT]
        #     elif all([self.data[FT][-1] <= self.data[FT][-2],
        #         self.data[FT][-2] <= self.data[FT][-3]]):
        #         desiredPositions[FT] = positionLimits[FT]
        #     else:
        #         desiredPositions[FT] = 0

        # # FT
        # if self.day == 0:
        #     desiredPositions[FT] = -positionLimits[FT]
        # elif self.day == 364:
        #     desiredPositions[FT] = 0
        # else:
        #     desiredPositions[FT] = self.positions[FT]



        # EMA
        trade_instruments = [PE, UD, GE]
        ema_periods = {PE:10, UD:12, FC:42, GE:14}  # EMA lookback period - TODO adjust for different instruments

        # We need enough data to calculate EMA
        for ins in trade_instruments:
            ema_period = ema_periods[ins]
            if self.day >= ema_period:
                # Decision logic based on price vs EMA relationship
                if self.data[ins][-1] < self.calculate_ema(self.data[ins][-ema_period:], ema_period):  # Price below EMA - buy
                    desiredPositions[ins] = positionLimits[ins]
                else:  # Price above EMA - short
                    desiredPositions[ins] = -positionLimits[ins]

            # For early days
            elif self.day >= 2:
                for ins in trade_instruments:
                    # Fallback to simple price comparison strategy
                    if self.data[ins][-2] > self.data[ins][-1]:
                        desiredPositions[ins] = positionLimits[ins]
                    else:
                        desiredPositions[ins] = -positionLimits[ins]



        # QUACK
        if self.day >= 2:
            if self.QUACKick:
                if self.day == self.QUACKStart + 22:
                    desiredPositions[QUACK] = -currentPositions[QUACK]
                elif self.QUACKick and self.day == self.QUACKStart + 44:
                    self.QUACKick = 0
                    desiredPositions[QUACK] = 0
                else:
                    desiredPositions[QUACK] = currentPositions[QUACK]

            else:
                if abs(self.data[QUACK][-1]-self.data[QUACK][-2]) > 0.05:
                    self.QUACKick = 1
                    desiredPositions[QUACK] = positionLimits[QUACK] if (self.data[QUACK][-1] > self.data[QUACK][-2]) else -positionLimits[QUACK]
                    self.QUACKStart = self.day
                else:
                    # Fallback to simple price comparison strategy
                    if self.data[QUACK][-2] > self.data[QUACK][-1]:
                        desiredPositions[QUACK] = positionLimits[QUACK]
                    else:
                        desiredPositions[QUACK] = -positionLimits[QUACK]






        # RW
        pattern, _ = self.rw_helper(self.data[RW])

        # down slope - short
        if self.day >= 2 and self.data[RW][-2] > self.data[RW][-1]:
            desiredPositions[RW] = -positionLimits[RW]
        else: # up slope - buy
            desiredPositions[RW] = positionLimits[RW]

        # # # THIS IS SHIT :/
        # # # sattle - buy
        # # if pattern == "potential_peak":
        # #     desiredPositions[RW] = positionLimits[RW]

        # # # peak - short
        # # if pattern == "potential_peak":
        # #     desiredPositions[RW] = -positionLimits[RW]

        # sudden drop - buy
        if pattern == "sudden_drop":
            desiredPositions[RW] = positionLimits[RW]

        # sudden rise - short
        if pattern == "sudden_rise":
            desiredPositions[RW] = -positionLimits[RW]






        # Display the end of trading day
        print("Ending Algorithm for Day:", self.day, "\n")
        #######################################################################
        # Return the desired positions
        return desiredPositions


    def rw_helper(self, prices, window_size=5, threshold=4):
        if len(prices) < window_size + 1:
            return None, 0  # Not enough data

        window = prices[-(window_size+1):-1]

        pattern = None
        confidence = 0


        if len(window) >= 3:
            # potential peak (price has been rising and now shows signs of reversal)
            trend = [prices[-i-1] < prices[-i] for i in range(1, 4)]
            if all(trend[:-1]) and not trend[-1]:
                # The price was rising but has started to decrease
                pattern = "potential_peak"
                confidence = (prices[-1] - min(window)) / min(window)

            # potential saddle (price has been falling and now shows signs of reversal)
            trend = [prices[-i-1] > prices[-i] for i in range(1, 4)]
            if all(trend[:-1]) and not trend[-1]:
                # The price was falling but has started to increase
                pattern = "potential_saddle"
                confidence = (max(window) - prices[-1]) / prices[-1]

        # sudden drop/ rise
        if len(prices) >= 2:
            if prices[-2] > 0:  # Avoid division by zero
                Δ = ((prices[-1] - prices[-2]) / prices[-2]) * 100
                if Δ < -threshold:
                    pattern = "sudden_drop"
                    confidence = abs(Δ) / threshold
                if Δ > threshold:
                    pattern = "sudden_rise"
                    confidence = abs(Δ) / threshold

        return pattern, confidence



    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average for a list of prices"""
        # Start with simple moving average
        sma = sum(prices[:period]) / period

        # Calculate multiplier
        multiplier = 2 / (period + 1)

        # Initialize EMA with SMA
        ema = sma

        # Calculate EMA
        for price in prices[period:]:
            ema = price * multiplier + ema * (1 - multiplier)

        return ema

