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

        self.lookback = 5
        self.z_threshold_short = .085
        self.z_threshold_long = .015

        self.fair_values = []
        self.z_scores = []
        self.performance = {'day': [], 'spread': [], 'z_score': [], 'signal': []}


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
        trade_instruments = [PE, UD, GE, DF]
        ema_periods = {PE:10, UD:12, FC:42, GE:14, DF:6}  # EMA lookback period - TODO adjust for different instruments

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
        fair_value = self.calculate_fair_value()
        actual_price = self.get_current_price(FC)

        # Calculate z-score
        z_score = self.calculate_z_score(actual_price, fair_value)

        # Trading logic based on z-score
        if z_score > self.z_threshold_short:
            # Fried Chicken is overpriced relative to model
            # Go short FC, long RC and SS (weighted by coefficients)
            self.performance['signal'].append(-1)  # Short signal

            # Short FC
            desiredPositions[FC] = -positionLimits[FC]

            # Long Raw Chicken - weighted by coefficient
            weight_rc = self.rc_coef / (self.rc_coef + self.ss_coef)
            desiredPositions[RC] = int(positionLimits[RC] * weight_rc)

            # Long Secret Spices - weighted by coefficient
            weight_ss = self.ss_coef / (self.rc_coef + self.ss_coef)
            desiredPositions[SS] = int(positionLimits[SS] * weight_ss)

        elif z_score < -self.z_threshold_long:
            # Fried Chicken is underpriced relative to model
            # Go long FC, short RC and SS (weighted by coefficients)
            self.performance['signal'].append(1)  # Long signal

            # Long FC
            desiredPositions[FC] = positionLimits[FC]

            # Short Raw Chicken - weighted by coefficient
            weight_rc = self.rc_coef / (self.rc_coef + self.ss_coef)
            desiredPositions[RC] = -int(positionLimits[RC] * weight_rc)

            # Short Secret Spices - weighted by coefficient
            weight_ss = self.ss_coef / (self.rc_coef + self.ss_coef)
            desiredPositions[SS] = -int(positionLimits[SS] * weight_ss)
        else:
            # No clear signal, stay neutral or unwind positions
            self.performance['signal'].append(0)  # Neutral signal

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

    # Calculate fair value of Fried Chicken based on regression model
    def calculate_fair_value(self):
        rc_price = self.get_current_price(RC)
        ss_price = self.get_current_price(SS)
        fair_value = self.intercept + (self.rc_coef * rc_price) + (self.ss_coef * ss_price)
        return fair_value

    # Calculate z-score for the spread between actual and fair value
    def calculate_z_score(self, actual_price, fair_value):
        # Need enough history to calculate meaningful z-scores
        if len(self.fair_values) < self.lookback:
            self.fair_values.append(fair_value)
            return 0

        self.fair_values.append(fair_value)
        # Keep only the most recent lookback period
        if len(self.fair_values) > self.lookback * 2:
            self.fair_values = self.fair_values[-self.lookback*2:]

        # Calculate spread between actual and fair value
        spread = actual_price - fair_value

        # Calculate recent spreads
        recent_spreads = [self.data[FC][-(i+1)] - self.fair_values[-(i+1)]
                         for i in range(min(self.lookback, len(self.fair_values)-1))]

        # Calculate mean and standard deviation of recent spreads
        mean_spread = np.mean(recent_spreads)
        std_spread = np.std(recent_spreads)

        # Avoid division by zero
        if std_spread == 0:
            return 0

        # Calculate z-score
        z_score = (spread - mean_spread) / std_spread
        self.z_scores.append(z_score)

        # Track performance metrics
        self.performance['day'].append(self.day)
        self.performance['spread'].append(spread)
        self.performance['z_score'].append(z_score)

        return z_score


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

