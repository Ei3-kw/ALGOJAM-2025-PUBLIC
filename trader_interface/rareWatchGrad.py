import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

def detect_peaks_and_saddles_gradient(prices, smoothing=True, window=15, poly_order=3, prominence=1.0):
    """
    Detect peaks and saddles using gradient analysis with optional smoothing
    
    Parameters:
    - prices: array of price values
    - smoothing: whether to apply Savitzky-Golay smoothing filter
    - window: window size for smoothing (must be odd)
    - poly_order: polynomial order for smoothing
    - prominence: minimum prominence for peak detection
    
    Returns:
    - peaks_idx: indices of peaks
    - saddles_idx: indices of saddles
    - smoothed_prices: smoothed price series (if smoothing=True)
    """
    # Apply smoothing if requested
    if smoothing:
        smoothed_prices = savgol_filter(prices, window, poly_order)
    else:
        smoothed_prices = prices
    
    # Compute gradient (first derivative)
    gradient = np.gradient(smoothed_prices)
    
    # Find peaks in original data (local maxima)
    peaks_idx, _ = find_peaks(smoothed_prices, prominence=prominence)
    
    # Find saddles in original data (local minima)
    # Invert the data to find minima as peaks
    saddles_idx, _ = find_peaks(-smoothed_prices, prominence=prominence)
    
    # Filter out false peaks and saddles based on gradient
    true_peaks = []
    for idx in peaks_idx:
        # Verify peak: gradient changes from positive to negative
        if idx > 0 and idx < len(gradient)-1:
            if gradient[idx-1] > 0 and gradient[idx+1] < 0:
                true_peaks.append(idx)
    
    true_saddles = []
    for idx in saddles_idx:
        # Verify saddle: gradient changes from negative to positive
        if idx > 0 and idx < len(gradient)-1:
            if gradient[idx-1] < 0 and gradient[idx+1] > 0:
                true_saddles.append(idx)
    
    return true_peaks, true_saddles, smoothed_prices, gradient

def detect_sudden_changes(prices, threshold=5.0):
    """
    Detect sudden price changes exceeding a percentage threshold
    
    Parameters:
    - prices: array of price values
    - threshold: minimum percentage change to be considered sudden
    
    Returns:
    - sudden_changes: list of (index, price, percentage_change, type)
    """
    sudden_changes = []
    
    for i in range(1, len(prices)):
        if prices[i-1] > 0:  # Avoid division by zero
            pct_change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
            if abs(pct_change) >= threshold:
                change_type = "rise" if pct_change > 0 else "drop"
                sudden_changes.append((i, prices[i], pct_change, change_type))
    
    return sudden_changes

def predict_next_movement(prices, peaks, saddles, gradient, window=10):
    """
    Predict next price movement based on recent trends and peak/saddle patterns
    
    Parameters:
    - prices: array of price values
    - peaks: indices of detected peaks
    - saddles: indices of detected saddles
    - gradient: array of gradient values
    - window: window of recent data to consider
    
    Returns:
    - prediction: string describing likely next movement
    - prediction_confidence: confidence level (0-1)
    """
    # Get recent data
    recent_prices = prices[-window:] if len(prices) >= window else prices
    recent_gradient = gradient[-window:] if len(gradient) >= window else gradient
    
    # Calculate trend direction and strength
    avg_gradient = np.mean(recent_gradient)
    gradient_strength = abs(avg_gradient)
    
    # Find most recent peak and saddle
    last_peak = max(peaks) if peaks else -1
    last_saddle = max(saddles) if saddles else -1
    
    # Determine most recent event
    last_event = "peak" if last_peak > last_saddle else "saddle" if last_saddle > -1 else "unknown"
    
    # Calculate distances to potential reversal points based on average cycle length
    peak_intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)] if len(peaks) >= 2 else [50]  # Default to 50 if not enough data
    saddle_intervals = [saddles[i+1] - saddles[i] for i in range(len(saddles)-1)] if len(saddles) >= 2 else [50]
    
    avg_cycle = np.mean(peak_intervals + saddle_intervals) if peak_intervals or saddle_intervals else 50
    
    # Make prediction
    prediction = ""
    confidence = 0.5  # Base confidence
    
    if last_event == "peak":
        days_since_peak = len(prices) - 1 - last_peak
        progress = days_since_peak / avg_cycle
        
        if avg_gradient < 0:
            # Downtrend after peak
            if progress < 0.5:
                prediction = "Continued decline from recent peak, approaching mid-cycle"
                confidence = 0.6 + min(0.3, gradient_strength)
            else:
                prediction = "Approaching a potential saddle formation"
                confidence = 0.5 + min(0.4, progress - 0.5)
        else:
            # Unusual uptrend after peak
            prediction = "Unexpected uptrend after peak, potential double-peak formation"
            confidence = 0.4
    
    elif last_event == "saddle":
        days_since_saddle = len(prices) - 1 - last_saddle
        progress = days_since_saddle / avg_cycle
        
        if avg_gradient > 0:
            # Uptrend after saddle
            if progress < 0.5:
                prediction = "Continued rise from recent saddle, approaching mid-cycle"
                confidence = 0.6 + min(0.3, gradient_strength)
            else:
                prediction = "Approaching a potential peak formation"
                confidence = 0.5 + min(0.4, progress - 0.5)
        else:
            # Unusual downtrend after saddle
            prediction = "Unexpected downtrend after saddle, potential double-bottom formation"
            confidence = 0.4
    
    else:
        # No clear cycle established yet
        if avg_gradient > 0:
            prediction = "Uptrend detected without established cycle, potential saddle formation in past"
        else:
            prediction = "Downtrend detected without established cycle, potential peak formation in past"
        confidence = 0.3
    
    return prediction, confidence

# Load data
data = pd.read_csv('data/Rare Watch_price_history.csv')
full_prices = data['Price'].values

for i in range(21, 365):
    print(i)
    prices = full_prices[0:i]

    # Detect peaks and saddles using gradient method
    peaks, saddles, smoothed_prices, gradient = detect_peaks_and_saddles_gradient(
        prices, smoothing=True, window=21, poly_order=3, prominence=2.0
    )

    # Detect sudden changes
    sudden_changes = detect_sudden_changes(prices, threshold=7.0)

    # Make prediction
    prediction, confidence = predict_next_movement(prices, peaks, saddles, gradient)

    # Plot results
    plt.figure(figsize=(14, 10))

    # Create subplots for better visualization
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Original Price', alpha=0.7)
    plt.plot(smoothed_prices, 'g-', label='Smoothed Price', linewidth=2)

    # Mark peaks and saddles
    for peak in peaks:
        plt.plot(peak, smoothed_prices[peak], 'r^', markersize=10, label='_nolegend_')
        plt.annotate(f'Day {peak}', (peak, smoothed_prices[peak]), xytext=(0, 10),
                     textcoords='offset points', ha='center')

    for saddle in saddles:
        plt.plot(saddle, smoothed_prices[saddle], 'gv', markersize=10, label='_nolegend_')
        plt.annotate(f'Day {saddle}', (saddle, smoothed_prices[saddle]), xytext=(0, -20),
                     textcoords='offset points', ha='center')

    # Mark sudden changes
    for idx, price, pct_change, change_type in sudden_changes:
        color = 'r' if change_type == 'drop' else 'g'
        marker = 'o'
        plt.plot(idx, price, marker=marker, color=color, markersize=8, label='_nolegend_')

    plt.title('Rare Watch Price with Gradient-Based Peak and Saddle Detection')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Plot gradient in second subplot
    plt.subplot(2, 1, 2)
    plt.plot(gradient, 'b-', label='Price Gradient')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Price Gradient (First Derivative)')
    plt.xlabel('Day')
    plt.ylabel('Gradient')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    # plt.show()

    # Generate report
    print("\nPeak and Saddle Detection Report")
    print("=" * 50)
    print(f"Total peaks detected: {len(peaks)}")
    print(f"Peak days: {peaks}")
    print(f"Total saddles detected: {len(saddles)}")
    print(f"Saddle days: {saddles}")
    print("\nSudden Price Changes")
    print("=" * 50)
    for idx, price, pct_change, change_type in sudden_changes:
        print(f"Day {idx}: {change_type.upper()} of {pct_change:.2f}% to ${price:.2f}")

    print("\nCycle Analysis")
    print("=" * 50)
    if len(peaks) >= 2:
        peak_intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        print(f"Average days between peaks: {np.mean(peak_intervals):.2f}")
    if len(saddles) >= 2:
        saddle_intervals = [saddles[i+1] - saddles[i] for i in range(len(saddles)-1)]
        print(f"Average days between saddles: {np.mean(saddle_intervals):.2f}")

    print("\nPrediction")
    print("=" * 50)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}")