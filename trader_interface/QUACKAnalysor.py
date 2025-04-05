import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# Load the data
data = pd.read_csv('data/Quantum Universal Algorithmic Currency Koin_price_history.csv')

# Basic stats and visualization
def analyze_price_data(data):
    # Plot the price over time
    plt.figure(figsize=(12, 6))
    plt.plot(data['Day'], data['Price'])
    plt.title('QUACK Price History')
    plt.xlabel('Day')
    plt.ylabel('Price ($)')
    plt.grid(True)
    
    # Calculate daily returns
    data['Return'] = data['Price'].pct_change() * 100
    
    # Plot the returns
    plt.figure(figsize=(12, 6))
    plt.plot(data['Day'][1:], data['Return'][1:])
    plt.title('QUACK Daily Returns')
    plt.xlabel('Day')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    print(f"Average price: ${data['Price'].mean():.2f}")
    print(f"Min price: ${data['Price'].min():.2f}")
    print(f"Max price: ${data['Price'].max():.2f}")
    print(f"Price volatility (std): ${data['Price'].std():.2f}")
    
    return data

# Autocorrelation analysis to identify periodicity
def autocorrelation_analysis(data, max_lag=100):
    prices = data['Price'].values
    
    # Calculate autocorrelation
    autocorr = np.correlate(prices - np.mean(prices), prices - np.mean(prices), mode='full')
    autocorr = autocorr[len(autocorr)//2:] # Take the second half
    autocorr = autocorr / autocorr[0] # Normalize
    
    # Plot autocorrelation
    plt.figure(figsize=(12, 6))
    plt.bar(range(min(max_lag, len(autocorr))), autocorr[:min(max_lag, len(autocorr))])
    plt.title('Autocorrelation Function')
    plt.xlabel('Lag (days)')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    
    # Find peaks in autocorrelation
    peaks, _ = signal.find_peaks(autocorr, height=0.2, distance=5)
    if len(peaks) > 0:
        print("\nPotential periods from autocorrelation peaks:")
        for i, peak in enumerate(peaks):
            if peak > 0:  # Skip the zero lag
                print(f"Peak {i+1}: {peak} days with autocorrelation of {autocorr[peak]:.2f}")
    else:
        print("\nNo significant autocorrelation peaks found.")
    
    return autocorr, peaks

# Spectral analysis to identify periodic components
def spectral_analysis(data):
    prices = data['Price'].values
    
    # Detrend the data
    detrended = prices - np.polyval(np.polyfit(np.arange(len(prices)), prices, 1), np.arange(len(prices)))
    
    # Compute FFT
    n = len(detrended)
    yf = fft(detrended)
    xf = fftfreq(n, 1)[:n//2]
    
    # Get the power spectrum (magnitude)
    power = 2.0/n * np.abs(yf[:n//2])
    
    # Plot the spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(xf[1:], power[1:]) # Skip the DC component
    plt.title('Power Spectrum')
    plt.xlabel('Frequency (cycles/day)')
    plt.ylabel('Power')
    plt.grid(True)
    
    # Find the top peaks in the spectrum
    peaks, _ = signal.find_peaks(power, height=0.01)
    peak_freqs = xf[peaks]
    peak_periods = 1 / peak_freqs
    peak_powers = power[peaks]
    
    # Sort by power
    sorted_indices = np.argsort(peak_powers)[::-1]
    top_periods = peak_periods[sorted_indices]
    top_powers = peak_powers[sorted_indices]
    
    if len(top_periods) > 0:
        print("\nTop periodic components from spectral analysis:")
        for i in range(min(5, len(top_periods))):
            if top_periods[i] < n/2:  # Only report periods shorter than half the data length
                print(f"Period {i+1}: {top_periods[i]:.1f} days with power of {top_powers[i]:.4f}")
    else:
        print("\nNo significant periodic components found.")
    
    return xf, power, top_periods, top_powers

# Seasonal decomposition
def seasonal_decomposition(data):
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Try to find the most likely period
        result = seasonal_decompose(data['Price'], model='additive', period=28)
        
        # Plot the decomposition
        plt.figure(figsize=(12, 10))
        
        plt.subplot(411)
        plt.plot(result.observed)
        plt.title('Observed')
        plt.grid(True)
        
        plt.subplot(412)
        plt.plot(result.trend)
        plt.title('Trend')
        plt.grid(True)
        
        plt.subplot(413)
        plt.plot(result.seasonal)
        plt.title('Seasonal')
        plt.grid(True)
        
        plt.subplot(414)
        plt.plot(result.resid)
        plt.title('Residual')
        plt.grid(True)
        
        plt.tight_layout()
        
        print("\nSeasonal decomposition completed with period=28")
        print("Note: This is a fixed period used for illustration. Adjust based on spectral analysis results.")
        
    except ImportError:
        print("statsmodels package not available for seasonal decomposition.")
    except Exception as e:
        print(f"Error in seasonal decomposition: {e}")

# Rolling statistics to identify trend changes
def rolling_analysis(data, window=14):
    data['MA'] = data['Price'].rolling(window=window).mean()
    data['Volatility'] = data['Price'].rolling(window=window).std()
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(data['Day'], data['Price'], label='Price')
    plt.plot(data['Day'], data['MA'], label=f'{window}-day MA', color='red')
    plt.title('Price and Moving Average')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(data['Day'][window-1:], data['Volatility'][window-1:], color='green')
    plt.title(f'{window}-day Rolling Volatility')
    plt.grid(True)
    
    plt.tight_layout()
    
    return data

def analyze_price_points(data, target_price=2.2, min_gap=10):
    # Find days where price equals the target price
    target_days = data[data['Price'] == target_price]['Day'].tolist()

    print(f"Found {len(target_days)} days where price = ${target_price}")

    # Calculate gaps between consecutive occurrences
    gaps = []
    for i in range(1, len(target_days)):
        gap = target_days[i] - target_days[i-1]
        if gap >= min_gap:
            print(f"START-{target_days[i-1]}, END-{target_days[i]}")
            gaps.append(gap)

    print(f"Number of significant gaps (>= {min_gap} days): {len(gaps)}")

    if gaps:
        # Basic statistics on gaps
        avg_gap = np.mean(gaps)
        median_gap = np.median(gaps)
        std_gap = np.std(gaps)

        print(f"Average gap: {avg_gap:.2f} days")
        print(f"Median gap: {median_gap:.2f} days")
        print(f"Standard deviation: {std_gap:.2f} days")

        # Count frequency of each gap length
        gap_counts = {}
        for gap in gaps:
            if gap in gap_counts:
                gap_counts[gap] += 1
            else:
                gap_counts[gap] = 1

        # Sort by gap length
        sorted_gaps = sorted(gap_counts.items())

        print("\nDistribution of gap lengths:")
        for gap, count in sorted_gaps:
            print(f"Gap of {gap} days: {count} occurrences ({count/len(gaps)*100:.1f}%)")

        # Find the most common gap(s)
        max_count = max(gap_counts.values())
        most_common_gaps = [gap for gap, count in gap_counts.items() if count == max_count]

        if len(most_common_gaps) == 1:
            print(f"\nMost common gap: {most_common_gaps[0]} days ({max_count} occurrences)")
        else:
            print(f"\nMost common gaps: {', '.join(map(str, most_common_gaps))} days ({max_count} occurrences each)")

        # Visualize the gaps
        plt.figure(figsize=(12, 6))

        # Plot histogram
        plt.subplot(1, 2, 1)
        plt.hist(gaps, bins=range(min(gaps), max(gaps)+2), alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of Gaps Between ${target_price} Price Points')
        plt.xlabel('Gap Length (days)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Plot the gap sequence
        plt.subplot(1, 2, 2)
        plt.plot(range(len(gaps)), gaps, marker='o', linestyle='-', color='green')
        plt.title('Sequence of Gap Lengths')
        plt.xlabel('Gap Number')
        plt.ylabel('Gap Length (days)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Mark the 2.2 price points on the original price chart
        plt.figure(figsize=(12, 6))
        plt.plot(data['Day'], data['Price'], color='blue', alpha=0.6)
        plt.scatter(target_days, [target_price] * len(target_days), color='red', s=50)
        plt.title(f'QUACK Price with ${target_price} Price Points Highlighted')
        plt.xlabel('Day')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)

        # Analyze if the gaps show periodicity
        if len(gaps) >= 3:
            # Check if the sequence repeats
            is_periodic = True
            period_length = None

            # Try to identify a repeating pattern in the gaps
            for pattern_length in range(1, len(gaps) // 2 + 1):
                pattern_matches = True
                for i in range(pattern_length, len(gaps), pattern_length):
                    if i + pattern_length > len(gaps):
                        # Compare partial pattern
                        remaining = len(gaps) - i
                        if gaps[i:i+remaining] != gaps[:remaining]:
                            pattern_matches = False
                            break
                    elif gaps[i:i+pattern_length] != gaps[:pattern_length]:
                        pattern_matches = False
                        break

                if pattern_matches:
                    period_length = pattern_length
                    break

            if period_length:
                print(f"\nDetected repeating pattern in gaps with period length of {period_length}")
                print(f"The pattern is: {gaps[:period_length]}")

                # Calculate the overall price cycle length based on this pattern
                cycle_days = sum(gaps[:period_length])
                print(f"This corresponds to a price cycle of approximately {cycle_days} days")
            else:
                # Check if the gaps are clustered around a central value
                if std_gap / avg_gap < 0.25:  # Low relative std dev suggests consistent gaps
                    print(f"\nGaps are consistently around {avg_gap:.1f} days (low variation)")
                    print(f"This suggests a price cycle of approximately {avg_gap:.1f} days")
                else:
                    # Check for other patterns
                    print("\nNo simple repeating pattern detected in the gaps")
                    print("Analyzing for more complex patterns...")

                    # Try autocorrelation on the gap sequence
                    if len(gaps) >= 10:
                        from scipy import signal

                        # Calculate autocorrelation
                        autocorr = np.correlate(gaps - np.mean(gaps), gaps - np.mean(gaps), mode='full')
                        autocorr = autocorr[len(autocorr)//2:] # Take the second half
                        autocorr = autocorr / autocorr[0] # Normalize

                        # Find peaks in autocorrelation
                        peaks, _ = signal.find_peaks(autocorr[1:], height=0.5)
                        if len(peaks) > 0:
                            print(f"Potential cycles in the gap sequence detected at lags: {peaks + 1}")
                        else:
                            print("No clear periodicity detected in the gap sequence itself")

        return gaps, target_days
    else:
        print("No significant gaps found.")
        return [], []

# Main analysis
def main():
    print("Analyzing QUACK price data for periodicity...\n")
    
    # Process the data
    data_with_returns = analyze_price_data(data)
    
    # Autocorrelation analysis
    print("\n--- Autocorrelation Analysis ---")
    autocorr, peaks = autocorrelation_analysis(data)
    
    # Spectral analysis
    print("\n--- Spectral Analysis ---")
    xf, power, periods, powers = spectral_analysis(data)
    
    # Seasonal decomposition
    print("\n--- Seasonal Decomposition ---")
    if len(periods) > 0:
        seasonal_decomposition(data)
    
    # Rolling analysis
    print("\n--- Rolling Statistics Analysis ---")
    rolling_analysis(data)
    
    # Conclusion
    print("\n--- Conclusion ---")
    if len(periods) > 0:
        main_period = periods[0]
        print(f"The primary cycle detected in the QUACK price is approximately {main_period:.1f} days.")
        print(f"Other potential cycles were detected at: {', '.join([f'{p:.1f} days' for p in periods[1:min(3, len(periods))]])}")
    else:
        print("No clear periodicity was detected in the QUACK price data.")
    
    # Main analysis
    print("Analyzing gaps between occurrences of price = $2.2...\n")
    gaps, target_days = analyze_price_points(data, target_price=2.2, min_gap=10)

    # Show if there are price patterns that aren't exactly 2.2
    print("\nChecking for near misses (prices very close to 2.2)...")
    near_misses = data[(data['Price'] > 2.195) & (data['Price'] < 2.205) & (data['Price'] != 2.2)]
    if not near_misses.empty:
        print(f"Found {len(near_misses)} days with prices very close to 2.2:")
        for _, row in near_misses.iterrows():
            print(f"Day {row['Day']}: ${row['Price']}")

    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()