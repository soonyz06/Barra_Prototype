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
    "STR": [["STR_21", "STR_10"], 1, -1],
    "MKT": [["MKT_3_5"], 1, 1],
    "BAB": [["BAB_3_5"], 1, -1]
}
categories = [c for c in pf_schema.keys() if c not in ["symbol", "ts"]]
composite_factors = list(factor_defs.keys())
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

def load_profile(symbols, FETCH=True):
    if not isinstance(symbols, list): symbols = [symbols]
    lf = loader.load_data(symbols, "Profile", fetch_func=loader.fetch_profile, schema=pf_schema, FETCH=FETCH)
    pf = lf.sort("ts", descending=False).unique(subset=["symbol"], keep="last").drop("ts")
    pf = pf.with_columns([pl.col(cat).fill_null("Unknown").alias(cat)for cat in categories])
    return pf
    
def load_asset_ret(symbols, FETCH=True):
    if not isinstance(symbols, list): symbols = [symbols]
    lf = loader.load_data(symbols, "History", fetch_func=loader.fetch_history, other_args={"period": "max"}, schema=his_schema, FETCH=FETCH)
    his = lf.sort("ts", descending=False).unique(subset=["symbol", "date"], keep="last")
    his = his.drop(['ts', 'open', 'high', 'low', 'volume', 'close', 'splits'])
    his = his.filter(pl.col("symbol").is_in(symbols))
    asset_ret = processor.log_transform(his.collect())
    return asset_ret #returns pl.LazyFrame

def construct_exposures(symbols, start_date, end_date, benchmark_symbol="MSCI", FETCH=True):
    symbols += [benchmark_symbol]
    
    #Loading Data
    pf = load_profile(symbols, FETCH)
    lf = load_asset_ret(symbols, FETCH)
    
    benchmark = (lf.filter(pl.col("symbol") == benchmark_symbol).select([pl.col("date"), pl.col("log_ret").alias("mkt_ret")]))
    lf = lf.filter(pl.col("symbol")!=benchmark_symbol)
    asset_ret = lf.collect()

    #Factor Construction
    for factors, unit, k in factor_defs.values():
        for factor in factors:
            val = factor.split("_") + [0]
            if val[0] == "MKT":
                lf = processor.add_mkt_beta(factor, lf, benchmark, int(val[1]), int(val[2]), 1)
            elif val[0] == "BAB":
                lf = processor.add_mkt_beta(factor, lf, benchmark, int(val[1]), int(val[2]), -1) #can reuse
                lf = processor.reverse_winsor(lf, [factor], p=0.33)
            else:
                lf = processor.add_log_change(factor, lf, int(val[1])*unit, int(val[2])*unit, k=k)
    
    for composite, (factors, _, _) in factor_defs.items():
        lf = processor.process_components(lf, factors, composite)

    #Factor Preprocessing
    lf = lf.filter((pl.col("date")>=start_date) & (pl.col("date")<=end_date))
    lf = lf.join(pf, on="symbol", how="left")
    lf = lf.select(["symbol", "date"]+composite_factors+sum(risk_factors.values(), []))
    lf = processor.process_composites(lf, composite_factors, risk_factors).sort(["symbol", "date"])
    
    lf = lf.select(["symbol", "date"]+composite_factors)
    lf_lagged = lf.with_columns(pl.all().exclude(["symbol", "date"]).shift(1).over("symbol")) #t-1
    exposures = lf_lagged.join(asset_ret.lazy(), on=["symbol", "date"], how="left")
    return exposures

def standard_t_stat(df, cols):
    #t = (x_bar - mu) / (std/sqrt(n)) = (y_bar * sqrt(n)) / std (mu=0) 
    t_stat = df.select([
        ((pl.col(c).mean() * pl.len().sqrt()) / pl.col(c).std()).alias(f"t-{c}")
        for c in cols
    ])
    return t_stat

def Newey_West_t_stat(df, cols):
    newey_west = {"cov_type": "HAC", "cov_kwds": {'maxlags': None}} #{} would be equivalent to standard t-stat
    
    X = np.ones(len(df))
    t_stats = {}
    for c in cols:
        model = sm.OLS(df[c].to_numpy(), X) #equivalent to fitting mean (β = Y_bar)
        results = model.fit(**newey_west) #can be used for CI and Durbin-Watson Stat
        t_stats[c] = results.tvalues[0] 
    return pl.DataFrame(t_stats)

def vol_targetting(df, cols, target_annual_vol_pct=None):
    #scale factor returns -> Beta makes more economical sense
    if target_annual_vol_pct is None: return df
    target_daily_vol = target_annual_vol_pct / np.sqrt(252)
    df = df.with_columns([
        (pl.col(col) * (target_daily_vol / pl.col(col).std())).alias(col)
        for col in cols
    ])
    return df

rng = np.random.default_rng(seed=42)
symbols = list(rng.choice(sp500_tickers, size=300, replace=False))
#symbols = sp500_tickers[:150]
exposures = construct_exposures(symbols, start_date=date(2015, 1, 1), end_date=date(2026, 1, 1), benchmark_symbol="SPY", FETCH=False)
factor_ret = processor.get_factor_returns(exposures, composite_factors, ["log_ret"]) ##Cross Sectional Regressions at each time step

factor_ret.write_parquet(basepath / "Test1.parquet")
plotter.plot_factor_performance(factor_ret, composite_factors)
#print(standard_t_stat(factor_ret, composite_factors))
print("\nNewey-West t-stat")
print(Newey_West_t_stat(factor_ret, composite_factors)) #?abs(t)>2

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
    result_df = pl.DataFrame({
        "Factor": X_cols,
        "Exposure": model.params,
        "Contribution": np.array(model.params) * np.array(f_sums),
        "Correlation": corrs
    }).with_columns([
        pl.col("Exposure").round(2),
        pl.col("Contribution").round(2),
        pl.col("Correlation").round(2)
    ])
    print(f"Performance Attribution: ({start_date} to {end_date})")
    print(model.summary(xname=X_cols)) 
    print(result_df)
    return model, result_df

def load_portfolio(symbols):
    print("\n", symbols)
    portfolio_df = load_asset_ret(symbols).collect()
    return portfolio_df.with_columns(
        (1 / pl.col("symbol").count().over("date")).alias("weight") #suppose equal weighting
    ).group_by("date").agg([
        ((pl.col("weight")*pl.col("log_ret")).sum()).alias("log_ret"), #log returns are additive
        pl.col("symbol").count().alias("n_assets")
    ])
        

#factor_ret = pl.read_parquet(basepath / "Test1.parquet") #could add ewma or smth, idk what im doing wrong
factor_ret = vol_targetting(factor_ret, composite_factors, target_annual_vol_pct=0.2) #betas make more economical sense

portfolio_df = load_portfolio(["SPY"])
model, result_df = decompose_returns(portfolio_df, factor_ret, date(2025, 1, 1), date(2026, 1, 1))

portfolio_df = load_portfolio(["IONQ", "HIMS", "QBTS", "OKLO", "RBLX", "PLTR"])
model, result_df = decompose_returns(portfolio_df, factor_ret, date(2025, 1, 1), date(2026, 1, 1))

portfolio_df = load_portfolio(["SRPT", "BMBL", "KHC", "MHO", "PDD"])
model, result_df = decompose_returns(portfolio_df, factor_ret, date(2025, 1, 1), date(2026, 1, 1))


# -----//Risk Decomposition//----     
"""
#Systematic Variance = w.T @ B @ omega @ B.T @ w
df = df.tail(252)
omega = np.cov(df.select(composite_factors).to_numpy().T)
#factor_exposures = B.T @ w
#systematic_variance = factor_exposures.T @ omega @ factor_exposures
print(omega)

#unit test, smooth/forward fill
"""








