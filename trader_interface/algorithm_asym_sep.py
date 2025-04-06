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

        # regression coefficients
        self.intercept = 1.6015
        self.rc_coef = 0.1675
        self.ss_coef = 0.0070

        # Generic parameters (placeholder - these are now split into specific ones)
        self.lookback = 5
        self.z_threshold_short = 0.02
        self.z_threshold_long = 0.02

        # Specific threshold parameters for different spread relationships
        # RC-FC relationship thresholds
        self.z_threshold_short_rc = 0.02  # For shorts when RC-FC spread is positive
        self.z_threshold_long_rc = 0.01   # For longs when RC-FC spread is negative

        # SS-FC relationship thresholds
        self.z_threshold_short_ss = 0.03  # For shorts when SS-FC spread is positive
        self.z_threshold_long_ss = 0.02   # For longs when SS-FC spread is negative

        # Performance tracking
        self.fair_values = []
        self.z_scores = []
        self.performance = {'day': [], 'spread': [], 'z_score': [], 'signal': []}

        # Separate tracking for individual relationships
        self.rc_performance = {'day': [], 'spread': [], 'z_score': [], 'signal': []}
        self.ss_performance = {'day': [], 'spread': [], 'z_score': [], 'signal': []}



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
        # trade_instruments = [RC, FC, SS]

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


        # Chickens
        # Calculate fair values for both relationships
        fair_value_rc = self.calculate_fair_value_rc()
        fair_value_ss = self.calculate_fair_value_ss()

        # Combined fair value (weighted average)
        fair_value = self.calculate_fair_value()

        # Get current price of FC
        actual_price = self.get_current_price(FC)

        # Calculate spreads
        spread_rc = actual_price - fair_value_rc
        spread_ss = actual_price - fair_value_ss
        spread = actual_price - fair_value

        # Update performance tracking for combined model
        self.performance['day'].append(self.day)
        self.performance['spread'].append(spread)

        # Update performance tracking for individual relationships
        self.rc_performance['day'].append(self.day)
        self.rc_performance['spread'].append(spread_rc)

        self.ss_performance['day'].append(self.day)
        self.ss_performance['spread'].append(spread_ss)

        # Calculate z-scores for each relationship
        z_score_rc = self.calculate_z_score(actual_price, fair_value_rc, self.rc_performance)
        z_score_ss = self.calculate_z_score(actual_price, fair_value_ss, self.ss_performance)

        # Calculate combined z-score
        z_score = self.calculate_z_score(actual_price, fair_value)

        self.performance['z_score'].append(z_score)
        self.rc_performance['z_score'].append(z_score_rc)
        self.ss_performance['z_score'].append(z_score_ss)

        # Combined signal based on both relationships
        # If both relationships suggest the same direction, take that signal
        # If they conflict, use the stronger signal (higher absolute z-score)

        # RC relationship signal
        rc_signal = 0
        if z_score_rc > self.z_threshold_short_rc:
            rc_signal = -1  # Short signal
        elif z_score_rc < -self.z_threshold_long_rc:
            rc_signal = 1   # Long signal

        # SS relationship signal
        ss_signal = 0
        if z_score_ss > self.z_threshold_short_ss:
            ss_signal = -1  # Short signal
        elif z_score_ss < -self.z_threshold_long_ss:
            ss_signal = 1   # Long signal

        # Record individual signals
        self.rc_performance['signal'].append(rc_signal)
        self.ss_performance['signal'].append(ss_signal)

        # Determine combined signal
        signal = 0

        # If both agree, use that signal
        if rc_signal == ss_signal and rc_signal != 0:
            signal = rc_signal
        # If they disagree, use the one with stronger conviction (higher abs z-score)
        elif rc_signal != 0 and ss_signal != 0:
            if abs(z_score_rc) > abs(z_score_ss):
                signal = rc_signal
            else:
                signal = ss_signal
        # If only one has a signal, use that
        elif rc_signal != 0:
            signal = rc_signal
        elif ss_signal != 0:
            signal = ss_signal

        self.performance['signal'].append(signal)

        # Trading logic based on combined signal
        if signal == -1:  # Short signal
            # Short FC
            desiredPositions[FC] = -positionLimits[FC]

            # Long Raw Chicken - weighted by coefficient
            weight_rc = self.rc_coef / (self.rc_coef + self.ss_coef)
            desiredPositions[RC] = int(positionLimits[RC] * weight_rc)

            # Long Secret Spices - weighted by coefficient
            weight_ss = self.ss_coef / (self.rc_coef + self.ss_coef)
            desiredPositions[SS] = int(positionLimits[SS] * weight_ss)

        elif signal == 1:  # Long signal
            # Long FC
            desiredPositions[FC] = positionLimits[FC]

            # Short Raw Chicken - weighted by coefficient
            weight_rc = self.rc_coef / (self.rc_coef + self.ss_coef)
            desiredPositions[RC] = -int(positionLimits[RC] * weight_rc)

            # Short Secret Spices - weighted by coefficient
            weight_ss = self.ss_coef / (self.rc_coef + self.ss_coef)
            desiredPositions[SS] = -int(positionLimits[SS] * weight_ss)

        else:  # Neutral signal
            # If we have existing positions, consider unwinding them gradually
            for instrument in [FC, RC, SS]:
                if currentPositions.get(instrument, 0) > 0:
                    # Reduce long positions by 50%
                    desiredPositions[instrument] = currentPositions[instrument] // 2
                elif currentPositions.get(instrument, 0) < 0:
                    # Reduce short positions by 50%
                    desiredPositions[instrument] = currentPositions[instrument] // 2


        # Display the end of trading day
        print("Ending Algorithm for Day:", self.day, "\n")
        #######################################################################
        # Return the desired positions
        return desiredPositions

    # Helper function to fetch the current price of an instrument
    def get_current_price(self, instrument):
        # return most recent price
        return self.data[instrument][-1]

    # Calculate fair value based on RC relationship
    def calculate_fair_value_rc(self):
        rc_price = self.get_current_price("Raw Chicken")
        return self.intercept + self.rc_coef * rc_price

    # Calculate fair value based on SS relationship
    def calculate_fair_value_ss(self):
        ss_price = self.get_current_price("Secret Spices")
        return self.intercept + self.ss_coef * ss_price

    # Combined fair value calculation (weighted average of both relationships)
    def calculate_fair_value(self):
        rc_fair_value = self.calculate_fair_value_rc()
        ss_fair_value = self.calculate_fair_value_ss()

        # Weight by coefficient magnitude (more weight to stronger relationship)
        total_coef = abs(self.rc_coef) + abs(self.ss_coef)
        weight_rc = abs(self.rc_coef) / total_coef
        weight_ss = abs(self.ss_coef) / total_coef

        return weight_rc * rc_fair_value + weight_ss * ss_fair_value

    # Calculate z-score using lookback period
    def calculate_z_score(self, actual, fair_value, performance_data=None):
        # Use main performance data if none specified
        if performance_data is None:
            performance_data = self.performance

        # Calculate spread
        spread = actual - fair_value

        # If not enough history, return 0
        if len(performance_data['spread']) < self.lookback:
            return 0

        # Calculate z-score based on recent spread history
        recent_spreads = performance_data['spread'][-self.lookback:]
        mean_spread = sum(recent_spreads) / len(recent_spreads)

        # Calculate standard deviation
        variance = sum((s - mean_spread) ** 2 for s in recent_spreads) / len(recent_spreads)
        std_spread = variance ** 0.5

        # Avoid division by zero
        if std_spread == 0:
            return 0

        return (spread - mean_spread) / std_spread


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

