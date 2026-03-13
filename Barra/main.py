from yahooquery import Ticker
import polars as pl
from datetime import datetime
from pathlib import Path
import re
from functools import reduce


class History:
    def __init__(self):
        self.basepath = Path.cwd() / "data" / "history"
        self.basepath.mkdir(exist_ok=True, parents=True)
        self.filepath = self.basepath / "history.parquet"

        self.identifiers = ["symbol", "date"]
        
    def fetch_history(self, symbols, period, interval="1d"):
        if len(symbols)==0: return None
        print(f"Fetching History ({period},{interval})-> {len(symbols)}")
        ticker = Ticker(symbols, asynchronous=True)
        history = ticker.history(period=period, interval=interval)
        if history.empty:
            print("Dataframe is empty")
            return None
        history = pl.from_pandas(history, include_index=True)
        history = history.drop("dividends")
        history = history.with_columns(pl.col("date").cast(pl.Date))
        return history

    def load_history(self, symbols):
        try:
            history = pl.read_parquet(self.filepath)
        except:
            history = self.fetch_history(symbols, period="max")
            history.write_parquet(self.filepath)
            return history

        missing = set(symbols) - set(history["symbol"].unique()) #unique is faster
        missing_df = self.fetch_history(missing, period="max")

        #date check + missing
        
        if missing_df is None or missing_df.is_empty():
            return history
        
        history = pl.concat([history, missing_df])
        history.write_parquet(self.filepath)
        return history

    def log_transform(self, df):
        return (
            df.sort(self.identifiers)
            .upsample(time_column="date", every="1d", group_by="symbol")
            .with_columns(
                pl.col("close").forward_fill().over("symbol") 
            )
            .lazy()
            .filter(pl.col("date").dt.weekday()<=5) #trading days only
            .with_columns(
                log_ret = pl.col("close").log().diff().over("symbol").fill_null(0)
            )
        )

    def add_log_change(self, factor, df, lookback_days, gap_days):
        #ln(a/b) = ln(a)-ln(b) (additive)
        #ln: 10->20 == 10-> 5 (symmetric)
        min_obs =  0.8
        window_days = int(lookback_days - gap_days)
        
        return (
            df.with_columns(
                pl.col("log_ret")
                .shift(gap_days) #pushed forward in time 
                .rolling_sum(
                    window_size=window_days,
                    weights=None,
                    min_samples=int(window_days*min_obs) #min_obs
                )
                .over("symbol")
                .alias(factor)            
            )
        )
                   
    def winsor_factor(self, df, factor, p=0.01):
        df = df.with_columns(
            pl.col(factor).clip(
                pl.col(factor).quantile(p).over("date"),
                pl.col(factor).quantile(1-p).over("date")
            ).alias(factor)
        )
        return df

    def znorm_factor(self, df, factor):
        df = df.with_columns([
            ((pl.col(factor)-pl.col(factor).mean().over("date")) /
             pl.col(factor).std().over("date"))
            .alias(factor)
        ])
        return df

    def combine_factors(self, df, factors, name, weights=None):
        assert isinstance(factors, list), "Should be a list"
        n = len(factors)
        if weights is None: weights = [1/n for i in range(n)]

        df = df.with_columns([
            pl.sum_horizontal([pl.col(f)*w for f, w in zip(factors, weights)])
        ]).alias(name)
        return self.znorm_factor(df, name)
        

symbols = ["AAPL", "META", "MSFT", "WMT", "COST", "NVDA"]


mo = 21 #trading days per month
his = History()
df = his.load_history(symbols)
df = his.log_transform(df)
df = his.add_log_change("UMD_12_1", df, 12*mo, mo)

df = df.collect()
print(df[["date", "log_ret", "close", "UMD_12_1"]].tail())

#Factor Neutralisation
#F=BX+e, where F is raw factor, X is risk factors and B is estimated
#zscore composure -> neut -> rescale (x-0)/sd: x/x.std().over("date")
#OLS equation :)


#high, low masking
#unit test, smooth forward fill
#sampling




