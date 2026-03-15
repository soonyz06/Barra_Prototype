import polars as pl
from datetime import datetime, date
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm

from src.loader import Loader
from src.processor import Processor
from src.plotter import Plots
from data.temp import sp500_tickers

pf_schema = {
    "symbol": pl.Utf8,
    "country": pl.Utf8,
    "industry": pl.Utf8,
    "sector": pl.Utf8,
    "ts": pl.Utf8
}

his_schema = {
    "symbol": pl.Utf8,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "adjclose": pl.Float64,
    "splits": pl.Float64,
    "ts": pl.Utf8
}
factor_defs = {
    "MOM": [["UMD_12_1", "UMD_6_1"], 21, 1], 
    "VAL": [["HML_5", "HML_3"], 21*12, -1],
    "STR": [["STR_21", "STR_10"], 1, -1]
}
categories = [c for c in pf_schema.keys() if c not in ["symbol", "ts"]]
composite_factors = list(factor_defs.keys())+["MKT"]
risk_factors = {"categorical": categories, "numerical": []}
#21 trading days per month

loader = Loader()
processor = Processor()
plotter = Plots()
#loader.compact_data("Profile", pf_schema)
#loader.compact_data("History", his_schema)


#-----//Factor Returns//-----
basepath = Path.cwd() / "data" / "Factor_Returns"
basepath.mkdir(exist_ok=True, parents=True)

def load_asset_ret(symbols):
    if not isinstance(symbols, list): symbols = [symbols]
    lf = loader.load_data(symbols, "History", fetch_func=loader.fetch_history, other_args={"period": "max"}, schema=his_schema)
    his = lf.sort("ts", descending=False).unique(subset=["symbol", "date"], keep="last")
    his = his.drop(['ts', 'open', 'high', 'low', 'volume', 'close', 'splits'])
    his = his.filter(pl.col("symbol").is_in(symbols))
    asset_ret = processor.log_transform(his.collect())
    return asset_ret #returns pl.LazyFrame

def construct_factor_returns(filename, symbols, start_date, end_date, benchmark_symbol="MSCI"):
    symbols += [benchmark_symbol]
    
    #Loading Data
    lf = loader.load_data(symbols, "Profile", fetch_func=loader.fetch_profile, schema=pf_schema)
    pf = lf.sort("ts", descending=False).unique(subset=["symbol"], keep="last").drop("ts")
    pf = pf.with_columns([pl.col(cat).fill_null("Unknown").alias(cat)for cat in categories])

    lf = load_asset_ret(symbols)
    benchmark = (lf.filter(pl.col("symbol") == benchmark_symbol).select([pl.col("date"), pl.col("log_ret").alias("mkt_ret")]))
    lf = lf.filter(pl.col("symbol")!=benchmark_symbol)
    asset_ret = lf.collect()

    #Factor Construction
    for factors, unit, k in factor_defs.values():
        for factor in factors:
            val = factor.split("_") + [0]
            lf = processor.add_log_change(factor, lf, int(val[1])*unit, int(val[2])*unit, k=k)
    for composite, (factors, _, _) in factor_defs.items():
        lf = processor.process_components(lf, factors, composite)
        
    lf = processor.add_mkt_beta(lf, benchmark)

    #Factor Preprocessing
    lf = lf.filter((pl.col("date")>=start_date) & (pl.col("date")<=end_date))

    lf = lf.join(pf, on="symbol", how="left")
    lf = lf.select(["symbol", "date"]+composite_factors+sum(risk_factors.values(), []))
    lf = processor.process_composites(lf, composite_factors, risk_factors).sort(["symbol", "date"])

    lf = lf.select(["symbol", "date"]+composite_factors)
    lf_lagged = lf.with_columns(pl.all().exclude(["symbol", "date"]).shift(1).over("symbol"))
    lf = lf_lagged.join(asset_ret.lazy(), on=["symbol", "date"], how="left")

    #Cross Sectional Regressions at each time step
    factor_ret = processor.get_factor_returns(lf, composite_factors, ["log_ret"])
    plotter.plot_factor_performance(factor_ret, composite_factors)
    factor_ret.write_parquet(basepath / f"{filename}.parquet")

rng = np.random.default_rng(seed=42)
symbols = list(rng.choice(sp500_tickers, size=100, replace=False))
construct_factor_returns("Test2", symbols, start_date=date(2020, 1, 1), end_date=date(2025, 1, 1), benchmark_symbol="SPY")


# -----//Performance Attribution//----- 
def decompose_returns(return_df, factor_ret, start_date, end_date): #could be used to decompose the returns of any asset, portfolio or manager's performance
    return_df = return_df.select([pl.col("date"), pl.col("log_ret").alias("asset_ret")])
    factor_ret = factor_ret.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date)).drop_nulls()
    factor_ret = factor_ret.join(return_df, on="date", how="left")
    
    factor_ret = factor_ret.with_columns(pl.lit(1.0).alias("alpha"))
    X_cols = composite_factors + ["alpha"]
    X = factor_ret.select(X_cols).to_numpy()
    y = factor_ret.select("asset_ret").to_numpy()
    model = sm.OLS(y, X).fit() 

    f_sums = [factor_ret[c].sum() if c != "alpha" else factor_ret.height for c in X_cols]
    corrs = [factor_ret.select(pl.corr("asset_ret", c)).item() for c in X_cols]
    attr_df = pl.DataFrame({
        "Factor": X_cols,
        "Exposure": model.params,
        "Contribution": np.array(model.params) * np.array(f_sums),
        "Correlation": corrs
    }).with_columns([
        pl.col("Exposure").round(2),
        pl.col("Contribution").round(2),
        pl.col("Correlation").round(2)
    ])
    print(f"\nPerformance Attribution: {symbol} ({start_date} to {end_date})")
    print(model.summary(xname=X_cols)) 
    print(attr_df)
    return attr_df

target_daily_vol = 0.20 / np.sqrt(252)
factor_ret = pl.read_parquet(basepath / "ret1.parquet")
if target_daily_vol is not None: #scale factor returns (vol-targetting) -> Beta makes economical sense
    factor_ret = factor_ret.with_columns([
        (pl.col(col) * (target_daily_vol / pl.col(col).std())).alias(col)
        for col in composite_factors
    ]) 

symbol = "BMBL"
return_df = load_asset_ret(symbol).collect()
result_df = decompose_returns(return_df, factor_ret, date(2025, 1, 1), date(2026, 1, 1))


# -----//Risk Decomposition//----     
"""
#Systematic Variance = w.T @ B @ omega @ B.T @ w
df = df.tail(252)
omega = np.cov(df.select(composite_factors).to_numpy().T)
#factor_exposures = B.T @ w
#systematic_variance = factor_exposures.T @ omega @ factor_exposures
print(omega)

#unit test, smooth forward fill
"""

#Risk decomposition, portfolio construction, scenario anaysis, hedging, etc
#Beta from OLS is ex post and Signal from factor construction is ex ante







