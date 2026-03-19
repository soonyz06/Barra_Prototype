import polars as pl
from datetime import datetime, date
from pathlib import Path
import time
import numpy as np
from scipy.stats import norm

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
                   
    def winsorisation(self, df, cols, p=0.01):
        return df.with_columns([
            pl.col(f).clip(
                pl.col(f).quantile(p).over("date"),
                pl.col(f).quantile(1-p).over("date")
            )
            .alias(f)
            for f in cols
        ])

    def combine_factors(self, df, factors, composite):
       return df.with_columns(
               pl.mean_horizontal(factors).alias(composite) #automatic reweighting (0.5*A + 0.5*B -> 1*A + 0*B when B is null), could also do sum_horizontal / used_weights
        ).drop(factors)

    def z_normalisation(self, df, cols):
        return df.with_columns([
            (pl.col(f) - pl.col(f).mean().over("date")) / (pl.col(f).std().over("date") + 1e-8)
            .alias(f)
            for f in cols
        ])

    def rescaling(self, df, cols):
        return df.with_columns([
            pl.col(f).fill_nan(0) / (pl.col(f).fill_nan(None).std().over("date") + 1e-8)
            .alias(f)
            for f in cols
        ])        

    def reverse_winsor(self, df, cols, p=0.05):
        return df.with_columns(
            pl.when(pl.col(f).is_between(
                pl.col(f).quantile(p).over("date"),
                pl.col(f).quantile(1-p).over("date"),
                closed="none")) #exclusive
            .then(pl.lit(None))
            .otherwise(pl.col(f))
            .alias(f)
            for f in cols
        )
    
    def median_imputation(self, df, cols): ##can improve when, what and how
        groups = [["date", "industry"], ["date", "sector"], ["date"]]
        return df.with_columns(
            pl.coalesce([pl.col(c), *[pl.col(c).median().over(g) for g in groups], pl.lit(0.0)]).alias(c)
            for c in cols
        )

    def one_hot_encoding(self, df, categories, drop_first=True):
        if isinstance(df, pl.LazyFrame): df = df.collect()
        dummy_df = df.select(categories).to_dummies(drop_first=drop_first) #drop_first=True so no perfect multicollinearity (non-invertible matrix)
        dummy_cols = dummy_df.columns
        return pl.concat([df, dummy_df], how="horizontal"), dummy_cols

    def train_regression(self, df, X_cols, y_cols, lambda_l2=0.0001):  ##add WLS by sqrt(MC)
        #Ridge Regression: (X'X + λI) β = X'Y (any intercepts are combined inside X)
        train_df = df.drop_nulls(subset=[*X_cols, *y_cols]) #df should alr be cleaned
        X = train_df.select(X_cols).to_numpy() #(n, K) 
        Y = train_df.select(y_cols).to_numpy() #(n, F)
        K = X.shape[1]
        
        LHS = X.T@X + lambda_l2*np.eye(K)
        RHS = X.T@Y
        beta = np.linalg.solve(LHS, RHS) #solves Ax=B 
        return beta #(K, F), regardless of n

    def minmax_scaling(self, df, cols):
        return df.with_columns([
            (pl.col(c) - pl.col(c).min().over("date")) / (pl.col(c).max().over("date") - pl.col(c).min().over("date")).alias(c)
            for c in cols
        ])

    def rank_normalisation(self, df, cols):
        return df.with_columns([
            ((pl.col(c).rank().over("date")-1) / (pl.col(c).count().over("date")-1)).alias(c)
            for c in cols
        ])

    def gaussian_rank_normalisation(self, df, cols):
        return df.with_columns([
            ((pl.col(c).rank().over("date") - 0.5) / pl.col(c).count().over("date")).map_batches(lambda x: norm.ppf(x)) #inverse CDF
            .alias(c) 
            for c in cols
        ])

    def get_residuals(self, df, X_cols, y_cols, beta):
        X = df.select(X_cols).to_numpy() #(N, K)
        Y = df.select(y_cols).to_numpy() #(N, F)
        residuals = Y - X@beta 
        return residuals #(N, F)

    def get_r2(self, df, X_cols, y_cols, betas):
        X = df.select(X_cols).to_numpy()
        Y = df.select(y_cols).to_numpy()
        y_hat = X @ betas
        residuals = Y - y_hat
        ssr = np.sum(residuals**2)
        sst = np.sum((Y - np.mean(Y))**2)
        r_squared = 1 - (ssr / sst)
        return r_squared
        
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
        lf = lf.with_columns(pl.lit(1.0).alias("intercept"))
        X_cols = X_cols + ["intercept"]
        
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
            .pipe(self.winsorisation, factors, p=0.01)
            .pipe(self.rank_normalisation, factors)
            .pipe(self.z_normalisation, factors)
            .pipe(self.combine_factors, factors, composite)
        )
    
    def process_composites(self, df, factors, risk_factors):
        return (
            df
            .pipe(self.z_normalisation, factors)
            .pipe(self.median_imputation, risk_factors["numerical"]) 
            .pipe(self.neutralise_factors, factors, risk_factors)
            .pipe(self.rescaling, factors)
            .pipe(self.median_imputation, factors)
        )

    def add_mkt_beta(self, factor, df, benchmark, vol, corr, k=1):
        min_obs = 0.7
        vol_days = vol * 252
        corr_days = corr * 252
        prior_b = 1
        w = 0.67
        
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
                (((w*(pl.col("corr") * (pl.col("asset_vol") / pl.col("mkt_vol"))) + (1-w)*prior_b)) * k).alias(factor) #bayesian shrinkage
            )
            .drop(["mkt_ret", "asset_vol", "mkt_vol", "corr"])
        )




    
