from yahooquery import Ticker
import polars as pl
from datetime import datetime
from pathlib import Path


class History:
    def __init__(self):
        self.basepath = Path.cwd() / "data" / "history"
        self.basepath.mkdir(exist_ok=True, parents=True)
        self.filepath = self.basepath / "history.parquet"
        
    def fetch_history(self, symbols):
        if len(symbols)==0: return None
        print(f"Fetching History -> {len(symbols)}")
        ticker = Ticker(symbols, asynchronous=True)
        history = ticker.history(period="1y", interval="1d")
        if history.empty:
            print("Dataframe is empty")
            return None
        history = pl.from_pandas(history, include_index=True)
        history = history.drop("dividends")
        return history

    def load_history(self, symbols):
        try:
            history = pl.read_parquet(self.filepath)
        except:
            history = self.fetch_history(symbols)
            history.write_parquet(self.filepath)

        missing = set(symbols) - set(history["symbol"].unique()) #unique is faster
        missing_df = self.fetch_history(missing)
        if missing_df is None or missing_df.is_empty():
            return history
        
        history = pl.concat([history, missing_df])
        history.write_parquet(self.filepath)
        return history

symbols = ["AAPL", "META", "MSFT"]



his = History()
df = his.load_history(symbols)
print(df.head())
print(set(df["symbol"].unique()))


#log returns are additive -> rolling sum of log returns







