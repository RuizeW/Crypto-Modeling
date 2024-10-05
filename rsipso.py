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
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to backtest the RSI strategy
def backtest_rsi(params, data):
    rsi_period, rsi_overbought, rsi_oversold = params
    data['RSI'] = calculate_rsi(data, period=int(rsi_period))

    # Initialize cash, position, and buy/sell signals
    initial_cash = 10000
    cash = initial_cash
    position = 0
    buy_signals = []
    sell_signals = []
    portfolio_values = []
    price_data = data['close'].values
    trade_count = 0  # Track the number of trades

    for i in range(len(data)):
        rsi = data['RSI'].iloc[i]
        price = data['close'].iloc[i]

        if np.isnan(rsi):
            buy_signals.append(None)
            sell_signals.append(None)
            portfolio_values.append(cash + position * price)
            continue

        if rsi < rsi_oversold and cash > 0:
            # Buy signal
            position = cash / price
            cash = 0
            buy_signals.append(price)
            sell_signals.append(None)
            trade_count += 1  # Record a trade
        elif rsi > rsi_overbought and position > 0:
            # Sell signal
            cash = position * price
            position = 0
            buy_signals.append(None)
            sell_signals.append(price)
            trade_count += 1  # Record a trade
        else:
            buy_signals.append(None)
            sell_signals.append(None)

        # Calculate the current portfolio value
        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)

    final_value = cash + position * price_data[-1]

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
    _, final_value, _ = backtest_rsi(params, data)
    return -final_value  # Since PSO minimizes the objective function, return negative final portfolio value

# Use PSO to optimize RSI parameters
def optimize_rsi(data):
    bounds = [(5, 30), (60, 80), (10, 30)]  # Bounds for RSI period, overbought, oversold parameters
    result = differential_evolution(objective_function, bounds, args=(data,), maxiter=100, disp=True)
    optimal_params = result.x
    final_value = -result.fun
    return optimal_params, final_value

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
    # Use uploaded file
    csv_file = 'D:/research project/data2024/1m.csv'

    # Read data
    data = read_data(csv_file)

    # Optimize RSI parameters using PSO
    optimal_params, final_value = optimize_rsi(data)
    print(f"Optimal RSI Parameters: {optimal_params}")
    print(f"Final Portfolio Value with optimized parameters: {final_value}")

    # Backtest and print trade count
    data, final_value, trade_count = backtest_rsi(optimal_params, data)
    print(f"Total Trades: {trade_count}")
    plot_results(data)

    # Calculate maximum drawdown
    max_drawdown = calculate_max_drawdown(data['portfolio_value'].values)
    print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

    # Plot performance and maximum drawdown
    plot_performance(data)
