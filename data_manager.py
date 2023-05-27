import yfinance as yf
import pandas as pd
import numpy as np


def create_data(tickers, start_date, end_date, assets_name):
    data = pd.DataFrame()

    print('Downloading finance data')
    for ticker in tickers:
        temp = yf.download(ticker, start=start_date, end=end_date, interval='1mo')
        data[ticker] = temp['Close']

    data['Bonds'] = 1.01**np.arange(len(data))  # monthly capitalization by 1% (simplification)
    data['Cash'] = 1    # value does not change (obvious lie  -> inflation)

    data.columns = assets_name

    return data
