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
        #Chose to use log ret instead of raw ret due to the following:
        #ln(a/b) = ln(a)-ln(b) ~ additive 
        #ln(20/10) + ln(5/10) = 0 ~ symmetry
        
        return (
            df.sort(self.identifiers).upsample(time_column="date", every="1d", group_by="symbol") #ensures trading day alignment
            .with_columns(
                pl.col("close").forward_fill().over("symbol") 
            )
            .lazy() 
            .filter(pl.col("date").dt.weekday()<=5) 
            .with_columns(
                log_ret = pl.col("close").log().diff().over("symbol").fill_null(0)
            )
        )

    def add_log_change(self, factor, df, lookback_days, gap_days=0, k=1):        
        min_obs =  0.8
        window_days = int(lookback_days - gap_days)
        
        return df.with_columns(
            (pl.col("log_ret")
             .shift(gap_days) #pushed forward in time 
             .rolling_sum( 
                 window_size=window_days,
                 weights=None, 
                 min_samples=int(window_days*min_obs) #min_obs validation
             )
             * k
            )
            .over("symbol")
            .alias(factor)            
        )
                   
    def winsor_factors(self, df, factors, p=0.01):
        return df.with_columns([
            pl.col(f).clip(
                pl.col(f).quantile(p).over("date"),
                pl.col(f).quantile(1-p).over("date")
            )
            .alias(f)
            for f in factors
        ])

    def combine_factors(self, df, factors, composite):
       return df.with_columns(
               pl.mean_horizontal(factors).alias(composite) #automatic reweighting (0.5*A + 0.5*B -> 1*A + 0*B when B is null), could also do sum_horizontal / used_weights
        ).drop(factors)

    def znorm_factors(self, df, factors):
        return df.with_columns([
            (pl.col(f) - pl.col(f).mean().over("date")) / (pl.col(f).std().over("date") + + 1e-8)
            .alias(f)
            for f in factors
        ])

    def neutralise_factors(self, df, factors):
        #keeps mu=0, so only need to rescale sd
        #Placeholder# #subtract mean? unwanted risk factors
        return df
    
    def rescale_factors(self, df, factors):
        return df.with_columns([
            pl.col(f) / (pl.col(f).std().over("date") + + 1e-8)
            .alias(f)
            for f in factors
        ])        

    def reverse_winsor(self, df, factor, p=0.05):
        low_mask = pl.col(factor).quantile(p).over("date")
        high_mask = pl.col(factor).quantile(1-p).over("date")
        
        return df.with_columns(
            pl.when(pl.col(factor).is_between(low_mask, high_mask, closed="none")) #exclusive
            .then(pl.lit(None))
            .otherwise(pl.col(factor))
            .alias(factor)
        )

    def process_factor(self, df, factors, composite):
        return (
            df
            .pipe(self.winsor_factors, factors, p=0.01)
            .pipe(self.znorm_factors, factors)
            .pipe(self.combine_factors, factors, composite)
            .pipe(self.znorm_factors, [composite])
            .pipe(self.neutralise_factors, [composite])
            .pipe(self.rescale_factors, [composite])
        )

symbols = ["AAPL", "META", "MSFT", "WMT", "COST", "NVDA"]
factor_defs = {
    "MOM": [["UMD_12_1", "UMD_6_1", "UMD_3_1"], 21, 1], #skips to ignore STR
    "VAL": [["HML_3", "HML_5"], 21*12, -1]
}
#21 trading days per month

his = History()
df = his.load_history(symbols)
df = his.log_transform(df) #returns lazy

for factors, unit, k in factor_defs.values():
    for factor in factors:
        val = factor.split("_") + [0]
        df = his.add_log_change(factor, df, int(val[1])*unit, int(val[2])*unit, k=k)

for composite, (factors, _, _) in factor_defs.items():
    df = his.process_factor(df, factors, composite)

df = df.collect()
print(df.columns)
print(df.tail())

#Factor Neutralisation
#F=BX+e, where F is raw factor, X is risk factors and B is estimated
#OLS equation :)

#unit test, smooth forward fill
#sampling




