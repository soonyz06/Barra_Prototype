not refined

# Data Pipeline
- Price-derived factors are temporarily used as a proxy for fundamental factors (Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and Momentum Everywhere. The Journal of Finance)
- Polars for lazy, partitioning, batch I/O operations, versioning, compact job, schema enforcement, etc
- Regressions: Time series, cross sectional, factor neutralisation/purification, etc

# Applications
- Hedging: Long BMBL -> unintended bet on VAL -> Long X BMBL and Short Y SRPT -> isolate idio alpha
- Risk decomposition, portfolio construction, performance attribution, scenario anaysis, etc
- MVO using factor returns instead of asset returns (nxn to kxk)
- OLS: F = (B'B)^-1 B'R ≈ Factor-mimicking portfolios: F = WR
- Beta from OLS is ex post and Signal from factor construction is ex ante

# Factor Returns
![me](img/Figure_1.png) 
![value](img/Figure_2.png)
- Differ quite a bit from actual values due to only small sample size, different 'definition' of factors, i haven't scaled by MC (WLS), i prob overlooked something in the logic, etc

# Performance Attribution
IONQ  
![ionq](img/IONQ.png)
  - IONQ's returns over the past year can be largely explained by its exposure to MOM and MKT factors

BMBL  
![bmbl](img/BMBL.png)

SRPT  
![srpt](img/SRPT.png)
- SRPT and BMBL both have exposures to VAL and MKT 


