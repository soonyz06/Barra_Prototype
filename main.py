import polars as pl
from datetime import datetime, date
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.loader import Loader
from data.temp import sp500_tickers

class Plots:
    def plot_null_heatmap(self, df):
        null_mask = df.select(pl.all().is_null()).to_pandas()
        plt.figure(figsize=(12, 8))
        sns.heatmap(null_mask, cbar=False, cmap="viridis")
        plt.title("Missing Data Heatmap (Yellow = Missing, Purple = Present)")
        plt.show()
        return self

    def plot_factor_performance(self, df, X_cols):
        pdf = df.to_pandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.set_index("date").sort_index()
        pdf.index = pd.DatetimeIndex(pdf.index)
        
        daily_rets = pdf[X_cols]
        cum_rets = daily_rets.cumsum()
        ann_ret = daily_rets.mean() * 252
        ann_vol = daily_rets.std() * np.sqrt(252)
        corrs = daily_rets.corr()

        fig1, ax_cum = plt.subplots(figsize=(14, 7))
        cum_rets.plot(ax=ax_cum, lw=2)
        ax_cum.set_title("Cumulative Factor Returns (1σ Tilt)", fontsize=14, fontweight='bold')
        ax_cum.set_ylabel("Log Ret")
        ax_cum.set_xlabel("Time")
        ax_cum.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

        fig2, (ax_corr, ax_table) = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(corrs, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_corr, cbar=False, center=0)
        ax_corr.set_title("Factor Correlation Matrix", fontweight='bold')
        ax_table.axis('off')
        stats_df = pd.DataFrame({
            "Avg. Return": ann_ret,
            "Avg. Vol": ann_vol,
        }).map(lambda x: f"{x:.2%}" if abs(x) < 1.0 else f"{x:.2f}")
        table = ax_table.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            rowLabels=stats_df.index,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        ax_table.set_title("Factor Summary", fontweight='bold')
        plt.tight_layout()
        plt.show()
        return self
        
class Processor:
    def __init__(self):
        self.basepath = Path.cwd() / "data" 
        self.basepath.mkdir(exist_ok=True, parents=True)
        self.identifiers = ["symbol", "date"]

    def log_transform(self, df):
        #Chose to use log ret instead of raw ret due to the following:
        #ln(a/b) = ln(a)-ln(b) ~ additive 
        #ln(20/10) + ln(5/10) = 0 ~ symmetry
        
        return (
            df.sort(self.identifiers).upsample(time_column="date", every="1d", group_by="symbol") #ensures trading day alignment
            .with_columns(
                pl.col("adjclose").forward_fill().over("symbol") 
            )
            .lazy() 
            .filter(pl.col("date").dt.weekday()<=5) 
            .with_columns(
                log_ret = pl.col("adjclose").log().diff().over("symbol").fill_null(0)
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
            (pl.col(f) - pl.col(f).mean().over("date")) / (pl.col(f).std().over("date") + 1e-8)
            .alias(f)
            for f in factors
        ])

    def rescale_factors(self, df, factors):
        return df.with_columns([
            pl.col(f).fill_nan(0) / (pl.col(f).fill_nan(None).std().over("date") + 1e-8)
            .alias(f)
            for f in factors
        ])        

    def reverse_winsor(self, df, factors, p=0.05):
        return df.with_columns(
            pl.when(pl.col(f).is_between(
                pl.col(f).quantile(p).over("date"),
                pl.col(f).quantile(1-p).over("date"),
                closed="none")) #exclusive
            .then(pl.lit(None))
            .otherwise(pl.col(f))
            .alias(f)
            for f in factors
        )
    
    def median_imputation(self, df, columns): ##can improve when, what and how
        groups = [["date", "industry"], ["date", "sector"], ["date"]]
        return df.with_columns(
            pl.coalesce([pl.col(c), *[pl.col(c).median().over(g) for g in groups], pl.lit(0.0)]).alias(c)
            for c in columns
        )

    def one_hot_encoding(self, df, categories, drop_first=True):
        if isinstance(df, pl.LazyFrame): df = df.collect()
        dummy_df = df.select(categories).to_dummies(drop_first=drop_first) #drop_first=True so no perfect multicollinearity (non-invertible matrix)
        dummy_cols = dummy_df.columns
        return pl.concat([df, dummy_df], how="horizontal"), dummy_cols

    def train_regression(self, df, X_cols, y_cols, lambda_l2=0.1):  ##add WLS by sqrt(MC)
        #Ridge Regression: (X'X + λI) β = X'Y (any intercepts are combined inside X)
        train_df = df.drop_nulls(subset=[*X_cols, *y_cols]) #df should alr be cleaned
        X = train_df.select(X_cols).to_numpy() #(n, K) 
        Y = train_df.select(y_cols).to_numpy() #(n, F)
        K = X.shape[1]
        
        LHS = X.T@X + lambda_l2*np.eye(K)
        RHS = X.T@Y
        beta = np.linalg.solve(LHS, RHS) #solves Ax=B 
        return beta #(K, F), regardless of n

    def get_residuals(self, df, X_cols, y_cols, beta):
        X = df.select(X_cols).to_numpy() #(N, K)
        Y = df.select(y_cols).to_numpy() #(N, F)
        residuals = Y - X@beta 
        return residuals #(N, F)
        
    def neutralise_factors(self, df, factors, risk_factors): #could do 3D linalg but my math ain't good enough
        #F= a + BX + e
        #,where F is the raw factor signal, X is exposures risk factors, B is the regression coefficients
        #,a to fit cross-sectional mean so that residual mu = 0 and and e is the 'pure' factor (X^T @ e = 0)
        df, risk_factors['dummies'] = self.one_hot_encoding(df, risk_factors['categorical'])
        lf = df.lazy().with_columns(pl.lit(1.0).alias("intercept"))
        X_cols = risk_factors['dummies']+risk_factors['numerical']+["intercept"] 
        y_cols = factors
        schema = lf.collect_schema()

        days = df.select(pl.col("date").n_unique()).item()
        symbols = df.select(pl.col("symbol").n_unique()).item()
        print(f"[INFO]Running cross-sectional regressions across {days:,} days and {symbols} assets: ")
        print(f"[INFO]Neutralising {len(factors)} factors against {len(risk_factors['categorical'])} categorical and {len(risk_factors['numerical'])} numerical risk factors")
        
        def _cross_sectional_regression(group_df):
            beta = self.train_regression(group_df, X_cols, y_cols)
            residuals = self.get_residuals(group_df, X_cols, y_cols, beta) #train on clean and regress on all
            return group_df.with_columns(
                pl.DataFrame(residuals, schema=y_cols)
            )
        return lf.group_by("date").map_groups(_cross_sectional_regression, schema=schema)

    def get_factor_returns(self, lf, X_cols, y_cols):
        schema = lf.select(["date", *X_cols]).collect_schema()
        
        def _cross_sectional_regression(group_df):
            betas = self.train_regression(group_df, X_cols, y_cols).flatten() #log returns per 1σ factor tilt
            return pl.DataFrame({
                "date": [group_df.select(pl.col("date").first()).item()], #fast
                **{col: [betas[i]] for i, col in enumerate(X_cols)}
            })
        
        result = lf.group_by("date").map_groups(_cross_sectional_regression, schema=schema).sort("date").collect()
        return result

    def process_components(self, df, factors, composite):
        return (
            df
            .pipe(self.winsor_factors, factors, p=0.01) 
            .pipe(self.znorm_factors, factors)
            .pipe(self.combine_factors, factors, composite)
        )
    def process_composites(self, df, factors, risk_factors):
        return (
            df.pipe(self.znorm_factors, factors)
            .pipe(self.median_imputation, risk_factors["numerical"]) 
            .pipe(self.neutralise_factors, factors, risk_factors)
            .pipe(self.rescale_factors, factors)
            .pipe(self.median_imputation, factors)
        )

    def add_mkt_beta(self, df, benchmark):
        min_obs = 0.7
        vol_days = 3 * 252
        corr_days = 5 * 252
        b = 1
        k = 0.33
        
        return (
            df.sort(["symbol", "date"])
            .join(benchmark, on="date", how="left")
            .with_columns([
                pl.col("log_ret").rolling_std(vol_days, min_samples=int(vol_days * min_obs))
                  .over("symbol").alias("asset_vol"),
                pl.col("mkt_ret").rolling_std(vol_days, min_samples=int(vol_days * min_obs))
                  .over("symbol").alias("mkt_vol"),
                pl.rolling_corr(pl.col("log_ret"), pl.col("mkt_ret"), window_size=corr_days, min_samples=int(corr_days * min_obs))
                  .over("symbol").alias("corr")
            ])
            .with_columns( #cov/vol equivalent to specific case of OLS
                MKT = (pl.col("corr") * (pl.col("asset_vol") / pl.col("mkt_vol"))) * (1-k) + k * b #bayesian shrinkage
            )
            .drop(["mkt_ret", "asset_vol", "mkt_vol", "corr"])
        )


#-----//Params//-----
start_date = date(2025, 1, 1)
end_date = date(2026, 3, 15)
rng = np.random.default_rng(seed=42)
benchmark_symbol = "SPY"
symbols = list(rng.choice(sp500_tickers, size=1, replace=False))#+[benchmark_symbol]

factor_defs = {
    "MOM": [["UMD_12_1", "UMD_6_1"], 21, 1], 
    "VAL": [["HML_3", "HML_5"], 21*12, -1],
    "STR": [["STR_21", "STR_10"], 1, -1]
}
#21 trading days per month

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

categories = [c for c in pf_schema.keys() if c not in ["symbol", "ts"]]
composite_factors = list(factor_defs.keys())+["MKT"]
risk_factors = {"categorical": categories, "numerical": []}


#-----//Main//-----
loader = Loader()
processor = Processor()
plotter = Plots()

if 1==1:
    #Loading Data
    lf = loader.load_data(symbols, "Profile", fetch_func=loader.fetch_profile, schema=pf_schema)
    pf = lf.sort("ts", descending=False).unique(subset=["symbol"], keep="last").drop("ts")
    pf = pf.with_columns([pl.col(cat).fill_null("Unknown").alias(cat)for cat in categories])
    #loader.compact_data("Profile", pf_schema)

    lf = loader.load_data(symbols, "History", fetch_func=loader.fetch_history, other_args={"period": "max"}, schema=his_schema)
    his = lf.sort("ts", descending=False).unique(subset=["symbol", "date"], keep="last")
    his = his.drop("ts").drop(['open', 'high', 'low', 'volume', 'close', 'splits'])
    #loader.compact_data("History", his_schema)

    lf = processor.log_transform(his.collect())
    benchmark = (lf.filter(pl.col("symbol") == benchmark_symbol).select([pl.col("date"), pl.col("log_ret").alias("mkt_ret")]))
    lf = lf.filter(pl.col("symbol")!=benchmark_symbol)
    ret = lf.collect()

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
    lf = lf_lagged.join(ret.lazy(), on=["symbol", "date"], how="left")

    #Factor Returns
    dirpath = Path.cwd() / "data" / "Factor_Returns"
    dirpath.mkdir(exist_ok=True, parents=True)
    
    factor_ret = processor.get_factor_returns(lf, composite_factors, ["log_ret"])
    #loader.write_data([factor_ret], dirpath)
    plotter.plot_factor_performance(factor_ret, composite_factors)

"""
dirpath = Path.cwd() / "data" / "Factor_Returns"
dirpath.mkdir(exist_ok=True, parents=True)
df = pl.scan_parquet(dirpath / "*.parquet").drop("ts").collect()
plotter.plot_factor_performance(df, composite_factors)
print(df.tail())
print(df.columns)
print(df.shape)

#Time-series regression for performance attribution (Ex-Post)

#Systematic Variance = w.T @ B @ omega @ B.T @ w
df = df.tail(252)
omega = np.cov(df.select(composite_factors).to_numpy().T)
#factor_exposures = B.T @ w
#systematic_variance = factor_exposures.T @ omega @ factor_exposures
print(omega)

#unit test, smooth forward fill
"""







