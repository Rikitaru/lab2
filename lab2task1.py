import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')

df_AAPL = pd.read_csv(r"D:\3\AAPL.csv", index_col = "Date")
df_INTC = pd.read_csv(r"D:\3\INTC.csv", index_col = "Date")
df_MA   = pd.read_csv(r"D:\3\MA.csv",   index_col = "Date")
df_V    = pd.read_csv(r"D:\3\V.csv",    index_col = "Date")
df_TSLA = pd.read_csv(r"D:\3\TSLA.csv", index_col = "Date")

win = 10
weights = np.arange(0.1, 0.6, 0.05)

def plot_graph(df, name):
    SMA10 = pd.DataFrame()
    SMA10['Adj Close'] = df['Adj Close'].rolling(window=10).mean()
    SMA15 = pd.DataFrame()
    SMA15['Adj Close'] = df['Adj Close'].rolling(window=15).mean()
    wma_company = df['Adj Close'].rolling(win).apply(
        lambda x: np.sum(weights * x) / weights.sum())

    plt.figure(figsize=(13, 5))
    plt.plot(df['Adj Close'], label='Adj Close Price')
    plt.plot(SMA10['Adj Close'], label='SMA10')
    plt.plot(SMA15['Adj Close'], label='SMA15')
    plt.plot(list(range(len(wma_company))), wma_company,
             'r--', label=f'WMA: window size = 10')
    plt.title(name)
    plt.xticks([])
    plt.xlabel('1 января 2021 - 1 января 2022')
    plt.ylabel('Отрегулированная цена в $')
    plt.legend(loc='upper right')
    plt.show()

plot_graph(df_AAPL,"APPLE")
plot_graph(df_INTC,"INTEL")
plot_graph(df_MA,"MASTERCARD")
plot_graph(df_V,"VISA")
plot_graph(df_TSLA,"TESLA")