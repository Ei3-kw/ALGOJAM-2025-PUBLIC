import numpy as np

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

        self.trailing_stop = {}
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
        
        trade_instruments = self.positionLimits.keys()

        # Display the prices of instruments I want to trade
        for ins in trade_instruments:
            print(f"{ins}: ${self.get_current_price(ins)}")


        # BENCH MARK -- 1409403

        # Start trading from Day 2 onwards. Buy if price dropped and sell if price rose compared to the previous day
        if self.day >= 2:
            for ins in trade_instruments:
                # if price has gone down buy
                if self.data[ins][-2] > self.data[ins][-1]:
                    desiredPositions[ins] = positionLimits[ins]
                else:
                    desiredPositions[ins] = -positionLimits[ins]

        # EMA
        trade_instruments = [PE, UD]
        ema_periods = {PE:10, UD:12, FC:25}  # EMA lookback period - TODO adjust for different instruments

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


        # SS

        # Strategy-specific params
        long_ema_period = 50
        short_ema_period = 15
        macd_signal_period = 9
        volatility_window = 20
        min_history = 60

        # State variables
        last_signal = None
        position_increased_at = None
        trailing_stop = {}

        ss_limit = positionLimits[SS]
        ss_price = self.get_current_price(SS)

        # For early days, use simple strategy until we have enough data
        if self.day < min_history:
            # Start with 50% position and gradually increase it
            desiredPositions[SS] = int(ss_limit * min(0.5, self.day / min_history))
        else:
            # We have enough data for more sophisticated analysis
            ss_prices = self.data[SS]

            # Calculate technical indicators
            rsi = self.calculate_rsi(ss_prices, 14)
            macd, signal = self.calculate_macd(ss_prices)
            volatility = self.calculate_volatility(
                ss_prices,
                volatility_window
            )
            trend_strength = self.calculate_trend_strength(
                ss_prices,
                long_ema_period,
                short_ema_period
            )

            # Long-term trend: Calculate 50-day EMA
            long_ema = self.calculate_ema(ss_prices, long_ema_period)

            # Short-term trend: Calculate 15-day EMA
            short_ema = self.calculate_ema(ss_prices, short_ema_period)

            # Current position
            current_position = currentPositions.get(SS, 0)

            # Position adjustment based on strategy
            if ss_price > long_ema:  # Confirm uptrend (price above long-term EMA)
                # Calculate base position based on trend strength
                base_position = int(ss_limit * min(0.8, 0.5 + trend_strength * 5))

                # Adjust for entry signals
                if short_ema > long_ema and macd is not None and signal is not None and macd > signal:
                    # Strong bullish signal - maximize position
                    desiredPositions[SS] = ss_limit
                    last_signal = "strong_bullish"
                elif rsi < 40:
                    # Oversold condition in bullish trend - good entry point
                    desiredPositions[SS] = ss_limit
                    last_signal = "oversold_bullish"
                else:
                    # Normal bullish trend - maintain base position
                    desiredPositions[SS] = base_position
                    last_signal = "bullish"

                # Update trailing stop loss
                self.set_trailing_stop(SS, ss_price, 0.12)  # 13% trailing stop

            else:  # Price below long-term EMA - potential downtrend
                if rsi > 65:
                    # Overbought in potential downtrend - reduce position
                    desiredPositions[SS] = int(ss_limit * 0.3)
                    last_signal = "overbought_caution"
                elif macd is not None and signal is not None and macd < signal:
                    # Bearish signal - reduce position even more
                    desiredPositions[SS] = int(ss_limit * 0.2)
                    last_signal = "bearish_signal"
                else:
                    # Unclear trend - maintain moderate position
                    desiredPositions[SS] = int(ss_limit * 0.5)
                    last_signal = "neutral"

            # Check if trailing stop was hit
            if current_position > 0 and self.check_trailing_stop(SS, ss_price):
                # Trailing stop hit - reduce position
                desiredPositions[SS] = int(ss_limit * 0.25)
                last_signal = "stop_loss_hit"

            # Volatility adjustment - lower position size during high volatility
            if volatility > 0.02:  # High volatility threshold
                # Scale back position proportionally to volatility
                volatility_factor = 1 - (volatility - 0.02) * 10  # Linear scaling
                volatility_factor = max(0.5, volatility_factor)  # Don't go below 50%
                desiredPositions[SS] = int(desiredPositions[SS] * volatility_factor)




        # QUACK
        # buy when less than avg - historically more likely to go up OR if price dropped
        if self.data[QUACK][-1] <= 2.2 or (self.day >= 2 and self.data[QUACK][-2] > self.data[QUACK][-1]):
            desiredPositions[QUACK] = positionLimits[QUACK]
        else: # short
             desiredPositions[QUACK] = -positionLimits[QUACK]

        # RW
        pattern, _ = self.rw_helper(self.data[RW])

        # down slope - short
        if self.day >= 2 and self.data[RW][-2] > self.data[RW][-1]:
            desiredPositions[RW] = -positionLimits[RW]
        else: # up slope - buy
             desiredPositions[RW] = positionLimits[RW]

        # # THIS IS SHIT :/
        # # sattle - buy
        # if pattern == "potential_peak":
        #     desiredPositions[RW] = positionLimits[RW]

        # # peak - short
        # if pattern == "potential_peak":
        #     desiredPositions[RW] = -positionLimits[RW]

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


    ########################################################
    # HELPER FUNCTIONS
    ########################################################
    def calculate_ema(self, prices, period):
        # Start with simple moving average
        sma = sum(prices[:period]) / period

        # Calculate multiplier
        multiplier = 2 / (period + 1)

        # Initialise EMA with SMA
        ema = sma

        # Calculate EMA
        for price in prices[period:]:
            ema = price * multiplier + ema * (1 - multiplier)

        return ema

    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        if len(prices) < max(fast_period, slow_period, signal_period):
            return None, None

        # Calculate fast and slow EMAs
        fast_ema = self.calculate_ema(prices[-fast_period*2:], fast_period)
        slow_ema = self.calculate_ema(prices[-slow_period*2:], slow_period)

        # MACD line is fast EMA - slow EMA
        macd = fast_ema - slow_ema

        # Use the last 'signal_period' values to calculate signal line
        if len(prices) >= slow_period + signal_period:
            # Calculate signal line (EMA of MACD)
            macd_values = []
            for i in range(signal_period):
                fast_ema_hist = self.calculate_ema(prices[-(i+1)-fast_period*2:-(i+1)], fast_period)
                slow_ema_hist = self.calculate_ema(prices[-(i+1)-slow_period*2:-(i+1)], slow_period)
                macd_values.insert(0, fast_ema_hist - slow_ema_hist)

            signal = self.calculate_ema(macd_values, signal_period)
            return macd, signal

        return macd, None

    def calculate_rsi(self, prices, period=13):
        # Relative Strength Index
        if len(prices) <= period:
            return 50  # not enough data

        # price changes
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100  # No losses

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    # def calculate_volatility(self, prices, window=20):
    #     # standard deviation of returns
    #     if len(prices) < window:
    #         return 0

    #     return np.std(np.diff(prices) / prices[:-1][-window:])

    def calculate_volatility(self, prices, window=20):
        if len(prices) < window + 1:  # Need at least window+1 points
            return 0

        # Take only the last window+1 prices
        recent_prices = prices[-(window+1):]
        # Calculate returns for the window
        returns = np.diff(recent_prices) / recent_prices[:-1]
        # Return standard deviation of the returns
        return np.std(returns)

    def calculate_trend_strength(self, prices, long_period=50, short_period=15):
        """Calculate trend strength using EMA relationship"""
        if len(prices) < long_period:
            return 0

        long_ema = self.calculate_ema(prices, long_period)
        short_ema = self.calculate_ema(prices, short_period)

        # trend strength as percentage gap between short and long EMAs
        return (short_ema - long_ema) / long_ema

    def set_trailing_stop(self, ins, price, stop_percentage=0.05):
        """Set trailing stop loss price"""
        if ins not in self.trailing_stop:
            self.trailing_stop[ins] = price * (1 - stop_percentage)
        else:
            # Update trailing stop if price moved higher
            new_stop = price * (1 - stop_percentage)
            if new_stop > self.trailing_stop[ins]:
                self.trailing_stop[ins] = new_stop

    def check_trailing_stop(self, ins, current_price):
        """Check if price hit trailing stop"""
        if ins in self.trailing_stop:
            return current_price < self.trailing_stop[ins]
        return False

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


