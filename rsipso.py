import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Function to read data
def read_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean().replace(0, np.nan).ffill()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Backtest function for the optimized RSI strategy
def backtest_rsi(params, data):
    rsi_period, rsi_overbought, rsi_oversold = params
    data = data.copy()

    data['RSI'] = calculate_rsi(data, period=int(rsi_period)).astype(float)

    # Initialize cash, position, and buy/sell signals
    initial_cash = 10000
    cash = initial_cash
    position = 0
    buy_signals = [None] * len(data)
    sell_signals = [None] * len(data)
    portfolio_values = [initial_cash] * len(data)
    trade_count = 0  # Track the number of trades

    for i in range(1, len(data)):
        rsi = data['RSI'].iloc[i - 1]  # Use previous day's RSI to avoid look-ahead bias
        price = data['close'].iloc[i]

        if np.isnan(rsi):
            portfolio_values[i] = cash + position * price
            continue

        if rsi < rsi_oversold and cash > 0:
            # Buy signal
            position = cash / price
            cash = 0
            buy_signals[i] = price
            trade_count += 1  # Record a trade
        elif rsi > rsi_overbought and position > 0:
            # Sell signal
            cash = position * price
            position = 0
            sell_signals[i] = price
            trade_count += 1  # Record a trade

        # Calculate the current portfolio value
        portfolio_values[i] = cash + position * price

    final_value = cash + position * data['close'].iloc[-1]

    # Append buy/sell signals and portfolio values to the data
    data['buy_signals'] = buy_signals
    data['sell_signals'] = sell_signals
    data['portfolio_value'] = portfolio_values

    return data, final_value, trade_count

# Function to calculate maximum drawdown
def calculate_max_drawdown(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown

# Objective function for optimization (maximize final portfolio value)
def objective_function(params, data):
    _, final_value, _ = backtest_rsi(params, data.copy())  # Use a copy of the data to avoid modifying the original
    return -final_value  # Since differential_evolution minimizes the objective function, return negative final portfolio value

# Use differential evolution to optimize RSI parameters
def optimize_rsi(data):
    bounds = [(5, 30), (60, 80), (10, 30)]  # Bounds for RSI period, overbought, oversold parameters
    result = differential_evolution(objective_function, bounds, args=(data,), maxiter=100, disp=False)
    optimal_params = result.x
    final_value = -result.fun
    return optimal_params, final_value

# Function to dynamically train and test the RSI strategy
def dynamic_train_test(data):
    initial_cash = 10000
    cash = initial_cash
    position = 0
    trade_count = 0
    window_size = 100  # Define a window size for training
    portfolio_values = []
    buy_signals = []
    sell_signals = []

    for i in range(window_size, len(data)):
        train_data = data.iloc[:i]  # Use data up to the current point as training data

        # Optimize RSI parameters using training data
        try:
            optimal_params, _ = optimize_rsi(train_data)
        except ValueError as e:
            print(f"Error optimizing parameters: {e}")
            price = data['close'].iloc[i]
            portfolio_values.append(cash + position * price)
            buy_signals.append(None)
            sell_signals.append(None)
            continue

        rsi_period, rsi_overbought, rsi_oversold = optimal_params

        # Calculate RSI for the current data point
        # Ensure there is enough data to calculate RSI
        if i - int(rsi_period) < 0:
            portfolio_values.append(cash + position * data['close'].iloc[i])
            buy_signals.append(None)
            sell_signals.append(None)
            continue

        rsi_data = data.iloc[i - int(rsi_period) + 1:i + 1]
        rsi = calculate_rsi(rsi_data, period=int(rsi_period)).iloc[-1]

        price = data['close'].iloc[i]

        if np.isnan(rsi):
            portfolio_values.append(cash + position * price)
            buy_signals.append(None)
            sell_signals.append(None)
            continue

        if rsi < rsi_oversold and cash > 0:
            # Buy signal
            position = cash / price
            cash = 0
            trade_count += 1
            buy_signals.append(price)
            sell_signals.append(None)
        elif rsi > rsi_overbought and position > 0:
            # Sell signal
            cash = position * price
            position = 0
            trade_count += 1
            buy_signals.append(None)
            sell_signals.append(price)
        else:
            buy_signals.append(None)
            sell_signals.append(None)

        # Record portfolio value
        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)

    # Create a new DataFrame for the testing period
    test_data = data.iloc[window_size:].copy()
    test_data['portfolio_value'] = portfolio_values
    test_data['buy_signals'] = buy_signals
    test_data['sell_signals'] = sell_signals

    final_value = cash + position * data['close'].iloc[-1]

    return test_data, final_value, trade_count

# Function to plot the backtest results
def plot_results(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data['close'], label='Price', color='black')
    plt.scatter(data.index, data['buy_signals'], label='Buy Signal', marker='^', color='green')
    plt.scatter(data.index, data['sell_signals'], label='Sell Signal', marker='v', color='red')
    plt.title('Backtest Results with Optimized RSI Strategy')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to plot performance and maximum drawdown
def plot_performance(data):
    # Plot portfolio value over time
    plt.figure(figsize=(10, 5))
    plt.plot(data['portfolio_value'], label='Portfolio Value', color='blue')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate and plot maximum drawdown
    portfolio_values = data['portfolio_value'].values
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, drawdown, label='Drawdown', color='red')
    plt.fill_between(data.index, drawdown, color='red', alpha=0.3)
    plt.title('Drawdown Over Time')
    plt.xlabel('Time')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main function
if __name__ == '__main__':
    # Use the uploaded file
    csv_file = 'D:/research project/data2024/m5.csv'

    # Read data
    data = read_data(csv_file)

    # Perform dynamic train and test
    data, final_value, trade_count = dynamic_train_test(data)
    print(f"Final Portfolio Value with dynamic training: {final_value}")
    print(f"Total Trades: {trade_count}")

    # Plot performance and maximum drawdown
    plot_performance(data)
