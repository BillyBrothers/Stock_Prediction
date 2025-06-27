import yfinance as yf

def load_stock_data(ticker: str, period: str ='max', interval: str = '1h'):
    """
    Load stock data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol. (e.g. "AAPL")
    period (str): Time range (e.g. "1y", "5d", "max")
    interval (str): Data interval (e.g. "1m", "5m", "1h", "1d")

    Returns:
    Dataframe: Historical OHLCV data with datetime index.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df