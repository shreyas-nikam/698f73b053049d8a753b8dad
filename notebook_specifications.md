
# Predicting Stock Beta & Factor Exposures: A CFA's Python Workflow for Performance Attribution and Risk Management

**Persona:** Alex, a CFA Charterholder and Portfolio Manager at "Alpha Investments."

**Scenario:** Alex is a seasoned Portfolio Manager at Alpha Investments, an asset management firm known for its data-driven investment strategies. While he's proficient with traditional financial analysis tools like Excel, his firm is pushing for more robust, reproducible, and scalable workflows using Python. Alex's current challenge is to deeply understand the risk and return drivers of the stocks in his actively managed equity portfolio. He needs to move beyond simply looking at market beta and delve into multi-factor models to truly attribute performance, manage systematic risks, and forecast returns under various economic conditions. This notebook will guide Alex through building a comprehensive factor model analysis from data acquisition to scenario planning, demonstrating how Python can transform his analytical capabilities.

---

## 1. Setup: Installing Libraries and Importing Dependencies

Alex begins by setting up his Python environment, ensuring all necessary libraries for data retrieval, statistical modeling, and visualization are installed and imported. This is a foundational step for any reproducible analysis.

```python
!pip install pandas numpy yfinance pandas-datareader statsmodels matplotlib seaborn scipy

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as pdr
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Configure matplotlib for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('deep')
```

---

## 2. Data Acquisition and Preparation: Building the Foundation

Alex needs to gather historical monthly total returns for his target stocks and the widely-used Fama-French factor returns, along with the risk-free rate. This process, often manual and error-prone in spreadsheets, is automated and made reproducible in Python. He'll select a diversified set of stocks to analyze and merge them with the Fama-French data.

The Fama-French factors, developed by Eugene Fama and Kenneth French, explain asset returns beyond just market risk.
The three factors are:
*   **Mkt-RF ($R_m - R_f$):** The excess return on the market portfolio over the risk-free rate. It captures market-wide systematic risk.
*   **SMB (Small Minus Big):** The return spread between portfolios of small-cap stocks and large-cap stocks. It captures the size effect.
*   **HML (High Minus Low):** The return spread between portfolios of high book-to-market (value) stocks and low book-to-market (growth) stocks. It captures the value effect.

Alex must ensure proper date alignment and unit conversion (Fama-French data is often in percentages, while `yfinance` returns are decimals) to avoid meaningless results.

```python
def retrieve_and_merge_data(tickers, start_date, end_date):
    """
    Retrieves stock returns from Yahoo Finance and Fama-French factor returns,
    then merges and prepares the data for factor model analysis.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date for data retrieval (e.g., '2014-01-01').
        end_date (str): End date for data retrieval (e.g., '2024-01-01').

    Returns:
        pd.DataFrame: Merged DataFrame with excess returns for stocks and
                      Fama-French factors (Mkt-RF, SMB, HML).
    """
    print("--- Retrieving Fama-French Factors ---")
    # Retrieve Fama-French 3-Factor Model data and Risk-Free Rate
    ff_data = pdr.DataReader('F-F_Research_Data_Factors', 'famafrench',
                             start=start_date, end=end_date)[0]
    
    # Convert Fama-French factors from percentage to decimal
    ff_data = ff_data / 100
    
    # Ensure index is datetime and aligned for merging
    ff_data.index = ff_data.index.to_timestamp()
    ff_data.index.name = 'Date'
    
    # Rename columns for clarity (Mkt-RF is already clear)
    ff_data = ff_data.rename(columns={'Mkt-RF': 'Mkt_RF'})

    print("--- Retrieving Stock Returns ---")
    # Retrieve stock adjusted close prices
    stock_prices = yf.download(tickers, start=start_date, end=end_date, interval='1mo')['Adj Close']
    
    # Calculate monthly percentage change
    stock_returns = stock_prices.pct_change().dropna()
    stock_returns.index.name = 'Date'

    print("--- Merging Data ---")
    # Merge stock returns with Fama-French factors
    # Use inner join to ensure only common dates are kept
    merged_data = stock_returns.join(ff_data, how='inner')
    
    # Compute excess returns for each stock
    for ticker in tickers:
        merged_data[f'{ticker}_excess'] = merged_data[ticker] - merged_data['RF']
    
    # Select relevant columns for analysis
    columns_for_analysis = [f'{t}_excess' for t in tickers] + ['Mkt_RF', 'SMB', 'HML'] + ['RF']
    merged_data = merged_data[columns_for_analysis].dropna()
    
    print(f"Merged dataset shape: {merged_data.shape[0]} months, {merged_data.shape[1]} columns.")
    return merged_data

# Define target stocks and date range
target_stocks = ['AAPL', 'BRK-B', 'TSLA', 'JNJ']
analysis_start_date = '2014-01-01'
analysis_end_date = '2024-01-01'

# Execute data retrieval and merging
df_merged = retrieve_and_merge_data(target_stocks, analysis_start_date, analysis_end_date)

# Display the head of the merged DataFrame
print("\nFirst 5 rows of the merged data:")
print(df_merged.head())

# Plot a sample to verify alignment (e.g., AAPL excess return vs Mkt-RF)
plt.figure(figsize=(12, 6))
plt.plot(df_merged.index, df_merged['AAPL_excess'], label='AAPL Excess Return')
plt.plot(df_merged.index, df_merged['Mkt_RF'], label='Market Excess Return (Mkt-RF)')
plt.title('AAPL Excess Return vs. Market Excess Return Over Time')
plt.xlabel('Date')
plt.ylabel('Monthly Return')
plt.legend()
plt.tight_layout()
plt.show()
```

Alex observes the initial rows of the merged dataset and a quick plot of `AAPL`'s excess returns against the market excess return. He visually confirms that the data is aligned and the returns generally co-move, suggesting a proper merge and conversion. The `Mkt_RF`, `SMB`, and `HML` factors represent the market, size, and value risk premiums, respectively, available for him to use in his models.

---

## 3. Establishing a Baseline: The Capital Asset Pricing Model (CAPM)

Before diving into complex multi-factor models, Alex starts with the fundamental Capital Asset Pricing Model (CAPM) to establish a baseline understanding of each stock's sensitivity to the overall market. The CAPM is a single-factor model that explains the expected return of an asset based on its market risk.

The CAPM equation is given by:
$$ R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \epsilon_{i,t} $$
Where:
*   $R_{i,t} - R_{f,t}$ is the excess return of asset $i$ at time $t$.
*   $\alpha_i$ (Jensen's Alpha) is the asset's abnormal return not explained by the market factor. A positive and statistically significant $\alpha_i$ indicates outperformance after adjusting for market risk.
*   $\beta_{i,M}$ (Market Beta) measures the asset's sensitivity to market movements. A $\beta_{i,M} > 1$ implies higher market sensitivity than the average stock, while $\beta_{i,M} < 1$ implies lower sensitivity.
*   $R_{m,t} - R_{f,t}$ is the market excess return at time $t$.
*   $\epsilon_{i,t}$ is the idiosyncratic error term.

Alex will perform this regression for each of his target stocks to obtain their individual market betas and Jensen's alpha, along with statistical significance.

```python
def run_capm_regression(df, stock_ticker):
    """
    Runs CAPM regression for a given stock and extracts key metrics.

    Args:
        df (pd.DataFrame): Merged DataFrame containing stock excess returns and Mkt_RF.
        stock_ticker (str): The ticker symbol of the stock to analyze.

    Returns:
        dict: A dictionary containing CAPM regression results.
    """
    y = df[f'{stock_ticker}_excess']
    X = sm.add_constant(df['Mkt_RF']) # Add constant for alpha
    
    model = sm.OLS(y, X).fit()
    
    alpha = model.params['const']
    beta_M = model.params['Mkt_RF']
    alpha_pval = model.pvalues['const']
    beta_M_pval = model.pvalues['Mkt_RF']
    r_squared = model.rsquared
    resid_std = model.resid.std()
    
    # Annualize alpha and residual standard deviation
    alpha_ann = alpha * 12
    resid_std_ann = resid_std * np.sqrt(12)
    
    # Calculate Information Ratio (if residual std is not zero)
    info_ratio = alpha_ann / resid_std_ann if resid_std_ann != 0 else np.nan

    return {
        'model': model,
        'alpha': alpha,
        'alpha_ann': alpha_ann,
        'alpha_pval': alpha_pval,
        'beta_M': beta_M,
        'beta_M_pval': beta_M_pval,
        'r_squared': r_squared,
        'resid_std_ann': resid_std_ann,
        'information_ratio': info_ratio
    }

# Store CAPM results for all stocks
capm_results = {}
for stock in target_stocks:
    print(f"\n--- CAPM Regression for {stock} ---")
    results = run_capm_regression(df_merged, stock)
    capm_results[stock] = results
    
    # Print summary table and key metrics
    print(results['model'].summary())
    print(f"\nCAPM Results for {stock}:")
    print(f"  Alpha (monthly): {results['alpha']:.4f} ({results['alpha_ann']:.2%} annualized)")
    print(f"  Alpha p-value: {results['alpha_pval']:.4f}")
    print(f"  Beta (Market): {results['beta_M']:.3f}")
    print(f"  Beta p-value: {results['beta_M_pval']:.4f}")
    print(f"  R-squared: {results['r_squared']:.3f}")
    print(f"  Annualized Residual Std Dev (Tracking Error): {results['resid_std_ann']:.3f}")
    print(f"  Information Ratio: {results['information_ratio']:.3f}")
```

Alex reviews the `statsmodels` output for each stock. He notes the market beta ($\beta_M$) values; for instance, a $\beta_M$ greater than 1, like for `TSLA`, suggests the stock is more volatile than the market. He also pays close attention to Jensen's Alpha ($\alpha$) and its p-value. A high p-value for alpha (e.g., > 0.05) indicates that the stock's abnormal return is not statistically significant and could be due to random chance rather than genuine skill. The R-squared value tells him the proportion of the stock's excess return variance explained by the market factor. For `AAPL`, an R-squared of around 0.3-0.5 is typical for a single stock, as idiosyncratic risk is significant.

---

## 4. Deeper Insights with Fama-French 3-Factor Model

While CAPM provides a basic understanding, Alex knows that investment performance is often driven by more than just market risk. He moves to the Fama-French 3-Factor Model, which adds size (SMB) and value (HML) factors, offering a richer explanation of asset returns and a more nuanced performance attribution.

The Fama-French 3-Factor Model equation is:
$$ R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \beta_{i,S}SMB_t + \beta_{i,H}HML_t + \epsilon_{i,t} $$
Where:
*   $\beta_{i,S}$ (Size Beta) measures the asset's exposure to the small-cap factor. A positive $\beta_{i,S}$ suggests a tilt towards smaller companies.
*   $\beta_{i,H}$ (Value Beta) measures the asset's exposure to the value factor. A positive $\beta_{i,H}$ suggests a tilt towards value stocks (high book-to-market), while a negative $\beta_{i,H}$ indicates a growth stock tilt (low book-to-market).
*   Other terms are as defined in the CAPM.

Alex will run this model for all stocks, compare their factor exposures (their "factor fingerprints"), and quantify the incremental explanatory power of the additional factors (SMB and HML). He'll also use the Information Ratio, defined as:
$$ IR = \frac{\hat{\alpha}_{\text{ann}}}{\hat{\sigma}_{\epsilon}\sqrt{12}} = \frac{\text{Annualized Alpha}}{\text{Annualized Tracking Error}} $$
The Information Ratio measures the risk-adjusted abnormal return, where $|IR| > 0.5$ is considered strong performance.

```python
def run_ff3_regression(df, stock_ticker):
    """
    Runs Fama-French 3-factor regression for a given stock and extracts key metrics.

    Args:
        df (pd.DataFrame): Merged DataFrame containing stock excess returns and FF factors.
        stock_ticker (str): The ticker symbol of the stock to analyze.

    Returns:
        dict: A dictionary containing FF3 regression results.
    """
    y = df[f'{stock_ticker}_excess']
    X = sm.add_constant(df[['Mkt_RF', 'SMB', 'HML']]) # Add constant for alpha
    
    model = sm.OLS(y, X).fit()
    
    alpha = model.params['const']
    beta_M = model.params['Mkt_RF']
    beta_S = model.params['SMB']
    beta_H = model.params['HML']
    
    alpha_pval = model.pvalues['const']
    
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_statistic = model.fvalue
    f_pvalue = model.f_pvalue
    
    resid_std = model.resid.std()
    
    # Annualize alpha and residual standard deviation
    alpha_ann = alpha * 12
    resid_std_ann = resid_std * np.sqrt(12)
    
    # Calculate Information Ratio (if residual std is not zero)
    info_ratio = alpha_ann / resid_std_ann if resid_std_ann != 0 else np.nan

    return {
        'model': model,
        'alpha': alpha,
        'alpha_ann': alpha_ann,
        'alpha_pval': alpha_pval,
        'beta_M': beta_M,
        'beta_S': beta_S,
        'beta_H': beta_H,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'resid_std_ann': resid_std_ann,
        'information_ratio': info_ratio
    }

# Store FF3 results for all stocks
ff3_results = {}
for stock in target_stocks:
    print(f"\n--- Fama-French 3-Factor Regression for {stock} ---")
    results = run_ff3_regression(df_merged, stock)
    ff3_results[stock] = results
    
    # Print summary table
    print(results['model'].summary())

# --- Comparative Analysis Table ---
print("\n--- Comparative Factor Exposure & Performance Table (CAPM vs. FF3) ---")
summary_data = []
for stock in target_stocks:
    capm_r = capm_results[stock]
    ff3_r = ff3_results[stock]
    
    summary_data.append({
        'Stock': stock,
        'CAPM_Alpha_Ann': f"{capm_r['alpha_ann']:.2%}",
        'CAPM_Alpha_pvalue': f"{capm_r['alpha_pval']:.3f}",
        'CAPM_Beta_M': f"{capm_r['beta_M']:.3f}",
        'CAPM_R_squared': f"{capm_r['r_squared']:.3f}",
        'CAPM_IR': f"{capm_r['information_ratio']:.3f}",
        'FF3_Alpha_Ann': f"{ff3_r['alpha_ann']:.2%}",
        'FF3_Alpha_pvalue': f"{ff3_r['alpha_pval']:.3f}",
        'FF3_Beta_M': f"{ff3_r['beta_M']:.3f}",
        'FF3_Beta_S': f"{ff3_r['beta_S']:.3f}",
        'FF3_Beta_H': f"{ff3_r['beta_H']:.3f}",
        'FF3_R_squared': f"{ff3_r['r_squared']:.3f}",
        'FF3_Adj_R_squared': f"{ff3_r['adj_r_squared']:.3f}",
        'FF3_IR': f"{ff3_r['information_ratio']:.3f}",
        'R_squared_Improvement': f"{(ff3_r['r_squared'] - capm_r['r_squared']):.3f}"
    })

df_summary = pd.DataFrame(summary_data).set_index('Stock')
print(df_summary)

# --- Factor Beta Comparison Bar Chart ---
betas_df = pd.DataFrame({
    'Stock': target_stocks,
    'Beta_M': [ff3_results[s]['beta_M'] for s in target_stocks],
    'Beta_S': [ff3_results[s]['beta_S'] for s in target_stocks],
    'Beta_H': [ff3_results[s]['beta_H'] for s in target_stocks]
})

betas_melted = betas_df.melt(id_vars='Stock', var_name='Factor', value_name='Beta')

plt.figure(figsize=(14, 7))
sns.barplot(x='Stock', y='Beta', hue='Factor', data=betas_melted)
plt.title('Fama-French 3-Factor Betas Comparison Across Stocks')
plt.xlabel('Stock')
plt.ylabel('Factor Beta')
plt.axhline(0, color='gray', linestyle='--')
plt.legend(title='Factor', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Security Market Line (SML) Plot ---
# Calculate average excess returns for SML plot
avg_excess_returns = df_merged[[f'{s}_excess' for s in target_stocks]].mean()

sml_data = []
for stock in target_stocks:
    sml_data.append({
        'Stock': stock,
        'Avg_Excess_Return': avg_excess_returns[f'{stock}_excess'],
        'Market_Beta': ff3_results[stock]['beta_M']
    })
df_sml = pd.DataFrame(sml_data)

# Theoretical SML
# Approximate market risk premium using the average Mkt_RF from our data
avg_mkt_rf = df_merged['Mkt_RF'].mean()
theoretical_sml_x = np.linspace(df_sml['Market_Beta'].min() * 0.8, df_sml['Market_Beta'].max() * 1.2, 100)
# Theoretical SML states E[R_i - R_f] = Beta_i * E[R_m - R_f]
# Annualize for visual comparison
theoretical_sml_y = theoretical_sml_x * avg_mkt_rf * 12

plt.figure(figsize=(12, 7))
sns.scatterplot(x='Market_Beta', y='Avg_Excess_Return', hue='Stock', data=df_sml.mul(12), s=100, zorder=2) # Annualize for plot
plt.plot(theoretical_sml_x, theoretical_sml_y, color='red', linestyle='--', label=f'Theoretical SML (E[Mkt-RF] Ann: {avg_mkt_rf*12:.2%})')
plt.title('Security Market Line (SML) Plot: Annualized Excess Returns vs. Market Beta')
plt.xlabel('Market Beta ($\\beta_M$)')
plt.ylabel('Annualized Average Excess Return')
plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(1, color='gray', linestyle=':', alpha=0.7, label='Market Beta = 1')
plt.legend()
plt.tight_layout()
plt.show()
```

Alex examines the `statsmodels` outputs and the comparative table. He notices how `TSLA` exhibits a high $\beta_M$ (market sensitivity) and a negative $\beta_H$ (growth tilt), consistent with a high-growth, high-volatility stock. `BRK-B`, on the other hand, likely has a more moderate $\beta_M$ and a positive $\beta_H$ (value tilt), reflecting its stable, value-oriented profile. The R-squared values for the FF3 model are generally higher than CAPM, indicating that size and value factors contribute to explaining stock returns, which is crucial for decomposing risk and attributing performance more accurately. He also checks the p-values for alpha again; a statistically insignificant alpha (p-value > 0.05) suggests that after accounting for market, size, and value factors, the stock did not generate abnormal returns. The Information Ratio provides a risk-adjusted measure of this alpha. The SML plot visually confirms if any stock delivered excess returns beyond what its market beta would suggest; stocks above the line have positive alpha.

---

## 5. Validating Model Assumptions: Diagnostic Tests

Before relying on the factor model for critical investment decisions, Alex must perform diagnostic tests to check if the underlying assumptions of Ordinary Least Squares (OLS) regression are met. Violations of these assumptions (e.g., autocorrelation, heteroskedasticity, multicollinearity) can lead to inefficient or biased parameter estimates and incorrect statistical inferences (e.g., t-statistics, p-values). This step ensures the robustness of his analysis.

He will check for:
*   **Autocorrelation (Durbin-Watson statistic):** Checks if residuals are correlated over time. For financial time series, positive autocorrelation (DW < 2, especially < 1.5) can indicate momentum effects or missing factors. The Durbin-Watson statistic is approximately $DW \approx 2(1 - \rho_1)$, where $\rho_1$ is the first-order autocorrelation of residuals. A value close to 2 indicates no autocorrelation.
*   **Heteroskedasticity (Breusch-Pagan test):** Checks if the variance of the residuals is constant across all levels of independent variables. Heteroskedasticity (Breusch-Pagan p-value < 0.05) leads to inefficient estimates and incorrect standard errors.
*   **Multicollinearity (Variance Inflation Factor - VIF):** Checks if independent variables are highly correlated with each other. High multicollinearity (VIF > 5 or 10) can make coefficient estimates unstable and difficult to interpret. The VIF for factor $j$ is given by $VIF_j = \frac{1}{1 - R_j^2}$, where $R_j^2$ is the R-squared from regressing factor $j$ on all other factors.

**Practitioner Warning:** Financial time-series data frequently exhibits heteroskedasticity and autocorrelation. If detected, Alex should consider using **Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors** (e.g., Newey-West) for more reliable statistical inference, as demonstrated in the code.

```python
def run_diagnostic_tests(model, X_df, stock_ticker):
    """
    Performs diagnostic tests for an OLS regression model.

    Args:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted OLS model.
        X_df (pd.DataFrame): DataFrame of independent variables (exogenous).
        stock_ticker (str): The ticker symbol of the stock for context.

    Returns:
        dict: A dictionary containing diagnostic test results.
    """
    # Durbin-Watson test for autocorrelation
    dw_stat = durbin_watson(model.resid)

    # Breusch-Pagan test for heteroskedasticity
    # model.model.exog refers to the original exogenous variables used in the model
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_pvalue = bp_test[1] # p-value from the Breusch-Pagan test

    # Variance Inflation Factor (VIF) for multicollinearity
    # Exclude the constant column for VIF calculation if it exists in X_df
    X_for_vif = X_df.drop(columns='const', errors='ignore')
    vif_results = {}
    if not X_for_vif.empty:
        for i, col in enumerate(X_for_vif.columns):
            vif = variance_inflation_factor(X_for_vif.values, i)
            vif_results[col] = vif
    else:
        vif_results['No_Factors'] = np.nan # For CAPM with only one factor, VIF is not typically computed against other factors

    # Interpret results
    dw_interpretation = "No significant autocorrelation (around 2)." if 1.5 <= dw_stat <= 2.5 else "Potential autocorrelation issues (outside 1.5-2.5 range)."
    bp_interpretation = "Homoskedasticity assumption likely holds (p-value >= 0.05)." if bp_pvalue >= 0.05 else "Heteroskedasticity detected (p-value < 0.05). Consider HAC standard errors."
    vif_interpretation = "No problematic multicollinearity (VIFs < 5)."
    if any(v > 5 for v in vif_results.values()):
        vif_interpretation = "Problematic multicollinearity detected (VIF > 5 for some factors)."

    return {
        'dw_stat': dw_stat,
        'dw_interpretation': dw_interpretation,
        'bp_pvalue': bp_pvalue,
        'bp_interpretation': bp_interpretation,
        'vif_results': vif_results,
        'vif_interpretation': vif_interpretation
    }

# Run diagnostics for each stock's FF3 model
diagnostic_results = {}
for stock in target_stocks:
    print(f"\n--- Diagnostic Tests for {stock}'s Fama-French 3-Factor Model ---")
    ff3_model = ff3_results[stock]['model']
    X_ff3 = sm.add_constant(df_merged[['Mkt_RF', 'SMB', 'HML']]) # Ensure X_df matches what was used in regression
    
    results = run_diagnostic_tests(ff3_model, X_ff3, stock)
    diagnostic_results[stock] = results

    print(f"  Durbin-Watson statistic: {results['dw_stat']:.3f} - {results['dw_interpretation']}")
    print(f"  Breusch-Pagan p-value: {results['bp_pvalue']:.4f} - {results['bp_interpretation']}")
    print(f"  VIF results: {results['vif_results']}")
    print(f"  VIF interpretation: {results['vif_interpretation']}")

    # --- Diagnostic 4-Panel Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Regression Diagnostic Plots for {stock} (Fama-French 3-Factor Model)', fontsize=16)

    # Residuals over time
    axes[0,0].plot(ff3_model.resid.index, ff3_model.resid)
    axes[0,0].axhline(y=0, color='red', linestyle='--')
    axes[0,0].set_title('Residuals Over Time')
    axes[0,0].set_ylabel('Residual Value')

    # Residuals vs fitted
    axes[0,1].scatter(ff3_model.fittedvalues, ff3_model.resid, alpha=0.5)
    axes[0,1].axhline(y=0, color='red', linestyle='--')
    axes[0,1].set_title('Residuals vs Fitted Values')
    axes[0,1].set_xlabel('Fitted Values')
    axes[0,1].set_ylabel('Residual Value')

    # Q-Q plot
    sm.qqplot(ff3_model.resid, line='45', ax=axes[1,0])
    axes[1,0].set_title('Q-Q Plot of Residuals')

    # Histogram of residuals
    axes[1,1].hist(ff3_model.resid, bins=30, edgecolor='black', alpha=0.7)
    axes[1,1].set_title('Residual Distribution')
    axes[1,1].set_xlabel('Residual Value')
    axes[1,1].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.show()
```

Alex reviews the diagnostic test results and the 4-panel plots for each stock. He notes that the Durbin-Watson statistic might be slightly below 2, suggesting some positive autocorrelation, which is common in financial time series (e.g., momentum in residuals). More critically, he often finds that the Breusch-Pagan p-value is less than 0.05, indicating the presence of heteroskedasticity. This means the standard errors and p-values from the basic OLS are unreliable. He makes a mental note that for real-world reporting, he would use **HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors**, such as Newey-West, to compute robust t-statistics and p-values, making his inferences valid despite these common financial data characteristics. He also checks VIFs, which are typically low for Fama-French factors (designed to be orthogonal), confirming no significant multicollinearity issues.

---

## 6. Dynamic Factor Exposures: Rolling Betas

Static, full-sample betas can mask how a stock's sensitivity to factors changes over time, especially during different market regimes or significant economic events. As a Portfolio Manager, Alex needs to understand this dynamic nature for effective risk management and tactical asset allocation. He will compute and visualize rolling betas over a defined window (e.g., 36 months) to observe how these exposures evolve.

This technique involves running the factor regression repeatedly on a moving window of historical data. The resulting time series of betas provides insights into how the stock's "factor fingerprint" adapts to changing market conditions.

```python
def calculate_rolling_betas(df, stock_ticker, window_size=36):
    """
    Calculates rolling Fama-French 3-factor betas for a given stock.

    Args:
        df (pd.DataFrame): Merged DataFrame with stock excess returns and FF factors.
        stock_ticker (str): The ticker symbol of the stock to analyze.
        window_size (int): The size of the rolling window in months.

    Returns:
        pd.DataFrame: DataFrame containing rolling betas (Beta_M, Beta_S, Beta_H).
    """
    rolling_betas = pd.DataFrame(index=df.index[window_size-1:],
                                 columns=['Beta_M', 'Beta_S', 'Beta_H'])
    
    y_col = f'{stock_ticker}_excess'
    X_cols = ['Mkt_RF', 'SMB', 'HML']

    for i in range(window_size - 1, len(df)):
        window_df = df.iloc[i - window_size + 1 : i + 1]
        
        y_roll = window_df[y_col]
        X_roll = sm.add_constant(window_df[X_cols])
        
        if len(y_roll) > len(X_cols) + 1: # Ensure enough observations for regression
            try:
                model_roll = sm.OLS(y_roll, X_roll).fit()
                rolling_betas.loc[df.index[i], ['Beta_M', 'Beta_S', 'Beta_H']] = model_roll.params[X_cols].values
            except ValueError:
                # Handle cases where regression might fail (e.g., singular matrix)
                rolling_betas.loc[df.index[i], ['Beta_M', 'Beta_S', 'Beta_H']] = np.nan
        else:
            rolling_betas.loc[df.index[i], ['Beta_M', 'Beta_S', 'Beta_H']] = np.nan

    return rolling_betas.astype(float)

# Define rolling window size
rolling_window = 36 # 36-month rolling window

# Calculate and plot rolling betas for each stock
for stock in target_stocks:
    print(f"\n--- Calculating Rolling Betas for {stock} (Window: {rolling_window} months) ---")
    rolling_betas_df = calculate_rolling_betas(df_merged, stock, rolling_window)
    
    plt.figure(figsize=(14, 7))
    rolling_betas_df.plot(ax=plt.gca())
    
    plt.title(f'Rolling {rolling_window}-Month Fama-French Factor Betas for {stock}')
    plt.xlabel('Date')
    plt.ylabel('Beta Value')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(1, color='gray', linestyle='--', alpha=0.5) # For market beta reference
    
    # Annotate significant market events
    # Dates are approximate and for illustrative purposes
    if datetime.datetime(2020, 2, 1) in rolling_betas_df.index.normalize() or \
       datetime.datetime(2020, 4, 1) in rolling_betas_df.index.normalize(): # Check if range overlaps
        plt.axvspan(datetime.datetime(2020, 2, 1), datetime.datetime(2020, 4, 1), 
                    color='red', alpha=0.2, label='COVID-19 Crash')
    if datetime.datetime(2022, 1, 1) in rolling_betas_df.index.normalize() or \
       datetime.datetime(2022, 12, 1) in rolling_betas_df.index.normalize(): # Check if range overlaps
        plt.axvspan(datetime.datetime(2022, 1, 1), datetime.datetime(2022, 12, 1), 
                    color='purple', alpha=0.1, label='Inflation/Rate Hikes')
    
    plt.legend(title='Factor', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
```

Alex analyzes the rolling beta plots. He observes how `AAPL`'s market beta ($\beta_M$) might increase during periods of market stress, like the COVID-19 crash, indicating it becomes more sensitive to market movements during downturns. Similarly, its growth tilt (negative $\beta_H$) might become more pronounced or fluctuate. This dynamic view of factor exposures is critical for him to understand the time-varying risk profile of his portfolio holdings. A stock that typically has a defensive beta might temporarily become more aggressive, or vice-versa, which could impact portfolio hedging strategies and risk budgeting.

---

## 7. Forward-Looking Analysis: Scenario Projections

One of the most powerful applications of factor models for Alex is to project expected returns under various hypothetical macroeconomic scenarios. This shifts the analysis from purely backward-looking performance attribution to a forward-looking risk management and strategic planning tool. By defining reasonable expected returns for the Fama-French factors in different economic environments, Alex can estimate how his target stocks might perform.

The scenario projection uses the estimated betas from the Fama-French 3-factor model:
$$ E[R_i - R_f] = \hat{\alpha}_i + \hat{\beta}_{i,M} E[R_m - R_f] + \hat{\beta}_{i,S} E[SMB] + \hat{\beta}_{i,H} E[HML] $$
Where $E[...]$ denotes the expected value of the factors under a specific scenario, and $\hat{\alpha}$, $\hat{\beta}$ are the estimated coefficients from the full-sample regression.

Alex will define several plausible scenarios and then calculate the projected annualized excess return for each stock.

```python
def project_returns_under_scenarios(ff3_model_params, scenarios):
    """
    Projects stock excess returns under defined macroeconomic scenarios.

    Args:
        ff3_model_params (pd.Series): Parameters (alpha, betas) from a fitted FF3 model.
        scenarios (dict): A dictionary where keys are scenario names and values are
                          dictionaries of expected factor returns (e.g., {'Mkt_RF': 0.08/12, ...}).

    Returns:
        pd.DataFrame: A DataFrame showing projected annualized excess returns for each stock under each scenario.
    """
    projected_returns_data = []
    
    # Extract alpha and betas from the model parameters
    alpha = ff3_model_params['const']
    beta_M = ff3_model_params['Mkt_RF']
    beta_S = ff3_model_params['SMB']
    beta_H = ff3_model_params['HML']

    for scenario_name, factor_returns in scenarios.items():
        # Calculate expected monthly excess return for the stock
        exp_ret_monthly = (alpha +
                           beta_M * factor_returns['Mkt_RF'] +
                           beta_S * factor_returns['SMB'] +
                           beta_H * factor_returns['HML'])
        
        # Annualize the expected excess return
        exp_ret_annual = exp_ret_monthly * 12
        projected_returns_data.append({'Scenario': scenario_name, 'Projected_Annual_Excess_Return': exp_ret_annual})

    return pd.DataFrame(projected_returns_data)


# Define hypothetical macroeconomic scenarios (monthly expected factor returns)
# These values are illustrative
macro_scenarios = {
    'Base Case': {'Mkt_RF': 0.08/12, 'SMB': 0.01/12, 'HML': 0.005/12},
    'Market Crash': {'Mkt_RF': -0.15/12, 'SMB': -0.05/12, 'HML': 0.02/12}, # Market down, flight to quality/value
    'Value Rotation': {'Mkt_RF': 0.01/12, 'SMB': -0.01/12, 'HML': 0.08/12}, # Market flat, value outperforms growth
    'Small-Cap Rally': {'Mkt_RF': 0.03/12, 'SMB': 0.06/12, 'HML': 0.00/12}, # Market up slightly, small caps strong
    'Stagflation': {'Mkt_RF': -0.05/12, 'SMB': -0.02/12, 'HML': 0.05/12} # Market down, growth down, value holds/gains
}

# Generate scenario projections for each stock
scenario_projections = {}
for stock in target_stocks:
    print(f"\n--- Scenario Projections for {stock} ---")
    ff3_model_params = ff3_results[stock]['model'].params
    projections = project_returns_under_scenarios(ff3_model_params, macro_scenarios)
    scenario_projections[stock] = projections
    print(f"Annualized Projected Excess Returns for {stock}:")
    print(projections.set_index('Scenario').applymap(lambda x: f"{x:.2%}"))

# Display all scenario projections in a single table for comparison
all_projections_df = pd.DataFrame()
for stock, projections in scenario_projections.items():
    projections_indexed = projections.set_index('Scenario').rename(columns={'Projected_Annual_Excess_Return': stock})
    if all_projections_df.empty:
        all_projections_df = projections_indexed
    else:
        all_projections_df = all_projections_df.join(projections_indexed)

print("\n--- Summary of Projected Annualized Excess Returns Across All Stocks ---")
print(all_projections_df.applymap(lambda x: f"{x:.2%}"))

```

Alex reviews the projected returns under different scenarios. He observes that in a 'Market Crash' scenario, `TSLA` (high market beta) shows a significantly larger projected negative return compared to `JNJ` (lower market beta). In a 'Value Rotation' scenario, `BRK-B` (positive value beta) might be projected to perform relatively better than `AAPL` or `TSLA` (negative value/growth tilt). This table provides Alex with critical insights for stress testing his portfolio, adjusting his risk exposure to specific factors, and informing his discussions with the investment committee about potential portfolio resilience or vulnerability to various economic outlooks.

---

## 8. Performance Attribution and Model Summary

To complete his comprehensive analysis, Alex wants to visualize the contribution of each factor to the cumulative excess return of a stock. This "cumulative return decomposition" helps him attribute performance to market, size, and value factors versus the stock's idiosyncratic alpha. He also wants a visual comparison of the model's predicted versus actual returns and a final summary of all key metrics for easy reporting.

The cumulative contribution of each factor at time $T$ is given by:
$$ \text{Cumulative Factor Contribution}_X = \sum_{t=1}^T \hat{\beta}_{X} \cdot F_{X,t} $$
where $F_{X,t}$ is the factor return for factor $X$ at time $t$. The cumulative alpha contribution is $\sum_{t=1}^T \hat{\alpha}$.

```python
def plot_cumulative_return_decomposition(df, ff3_model_params, stock_ticker):
    """
    Plots the cumulative return decomposition for a given stock.

    Args:
        df (pd.DataFrame): Merged DataFrame with stock excess returns and FF factors.
        ff3_model_params (pd.Series): Parameters (alpha, betas) from a fitted FF3 model.
        stock_ticker (str): The ticker symbol of the stock to analyze.
    """
    y = df[f'{stock_ticker}_excess']
    X = df[['Mkt_RF', 'SMB', 'HML']]

    alpha = ff3_model_params['const']
    beta_M = ff3_model_params['Mkt_RF']
    beta_S = ff3_model_params['SMB']
    beta_H = ff3_model_params['HML']

    # Calculate factor contributions
    market_contribution = beta_M * X['Mkt_RF']
    smb_contribution = beta_S * X['SMB']
    hml_contribution = beta_H * X['HML']
    alpha_contribution = pd.Series(alpha, index=df.index)

    # Sum of factor contributions + alpha should approximate actual excess return
    total_model_return = market_contribution + smb_contribution + hml_contribution + alpha_contribution

    # Residuals are the unexplained part (epsilon)
    epsilon_contribution = y - total_model_return

    # Create a DataFrame for plotting cumulative contributions
    attribution_df = pd.DataFrame({
        'Market Factor': market_contribution.cumsum(),
        'SMB Factor': smb_contribution.cumsum(),
        'HML Factor': hml_contribution.cumsum(),
        'Alpha': alpha_contribution.cumsum(),
        'Residual (Unexplained)': epsilon_contribution.cumsum(),
        'Actual Excess Return': y.cumsum()
    }, index=df.index)

    plt.figure(figsize=(14, 7))
    attribution_df[['Market Factor', 'SMB Factor', 'HML Factor', 'Alpha', 'Residual (Unexplained)']].plot(
        kind='area', stacked=True, ax=plt.gca(), alpha=0.7
    )
    attribution_df['Actual Excess Return'].plot(ax=plt.gca(), color='black', linestyle='--', linewidth=2, label='Actual Excess Return')

    plt.title(f'Cumulative Return Decomposition for {stock} (Fama-French 3-Factor Model)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_predicted_vs_actual(df, ff3_model, stock_ticker):
    """
    Plots predicted vs. actual excess returns with a 45-degree line.

    Args:
        df (pd.DataFrame): Merged DataFrame with stock excess returns and FF factors.
        ff3_model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted FF3 model.
        stock_ticker (str): The ticker symbol of the stock to analyze.
    """
    y_actual = df[f'{stock_ticker}_excess']
    y_predicted = ff3_model.predict(sm.add_constant(df[['Mkt_RF', 'SMB', 'HML']]))

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_predicted, y=y_actual, alpha=0.6)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 
             color='red', linestyle='--', label='45-degree line (Perfect Prediction)')
    plt.title(f'Predicted vs. Actual Excess Returns for {stock_ticker}')
    plt.xlabel('Predicted Excess Return')
    plt.ylabel('Actual Excess Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Plotting for each stock ---
for stock in target_stocks:
    ff3_model = ff3_results[stock]['model']
    ff3_model_params = ff3_model.params
    
    print(f"\n--- Visualizing Performance for {stock} ---")
    plot_cumulative_return_decomposition(df_merged, ff3_model_params, stock)
    plot_predicted_vs_actual(df_merged, ff3_model, stock)

# --- Final Comparative Summary Table ---
print("\n--- Comprehensive Factor Exposure & Performance Report ---")
final_summary_data = []
for stock in target_stocks:
    capm_r = capm_results[stock]
    ff3_r = ff3_results[stock]
    diag_r = diagnostic_results[stock]
    
    # Calculate R-squared improvement
    r_squared_improvement = ff3_r['r_squared'] - capm_r['r_squared']

    # VIFs for Mkt_RF, SMB, HML (from diagnostic_results)
    vifs_str = ", ".join([f"{k}: {v:.2f}" for k, v in diag_r['vif_results'].items()])
    
    final_summary_data.append({
        'Stock': stock,
        'FF3_Alpha_Ann (%)': ff3_r['alpha_ann'] * 100,
        'FF3_Alpha_pvalue': ff3_r['alpha_pval'],
        'FF3_Beta_M': ff3_r['beta_M'],
        'FF3_Beta_S': ff3_r['beta_S'],
        'FF3_Beta_H': ff3_r['beta_H'],
        'FF3_R_squared': ff3_r['r_squared'],
        'R2_Improvement (FF3-CAPM)': r_squared_improvement,
        'FF3_IR': ff3_r['information_ratio'],
        'DW_Stat': diag_r['dw_stat'],
        'BP_Pvalue': diag_r['bp_pvalue'],
        'VIFs': vifs_str
    })

df_final_report = pd.DataFrame(final_summary_data).set_index('Stock')
# Format for better readability in markdown table
df_final_report_formatted = df_final_report.copy()
df_final_report_formatted['FF3_Alpha_Ann (%)'] = df_final_report_formatted['FF3_Alpha_Ann (%)'].map('{:.2f}%'.format)
df_final_report_formatted['FF3_Alpha_pvalue'] = df_final_report_formatted['FF3_Alpha_pvalue'].map('{:.3f}'.format)
df_final_report_formatted['FF3_Beta_M'] = df_final_report_formatted['FF3_Beta_M'].map('{:.3f}'.format)
df_final_report_formatted['FF3_Beta_S'] = df_final_report_formatted['FF3_Beta_S'].map('{:.3f}'.format)
df_final_report_formatted['FF3_Beta_H'] = df_final_report_formatted['FF3_Beta_H'].map('{:.3f}'.format)
df_final_report_formatted['FF3_R_squared'] = df_final_report_formatted['FF3_R_squared'].map('{:.3f}'.format)
df_final_report_formatted['R2_Improvement (FF3-CAPM)'] = df_final_report_formatted['R2_Improvement (FF3-CAPM)'].map('{:.3f}'.format)
df_final_report_formatted['FF3_IR'] = df_final_report_formatted['FF3_IR'].map('{:.3f}'.format)
df_final_report_formatted['DW_Stat'] = df_final_report_formatted['DW_Stat'].map('{:.3f}'.format)
df_final_report_formatted['BP_Pvalue'] = df_final_report_formatted['BP_Pvalue'].map('{:.4f}'.format)

print(df_final_report_formatted.to_markdown())
```

Alex examines the cumulative return decomposition plots. For `TSLA`, he might see that the 'Market Factor' contributes significantly to its cumulative return, but there's also a noticeable 'Alpha' component or 'Residual (Unexplained)' component that the model doesn't capture. For `BRK-B`, the 'HML Factor' (value) might show a more consistent positive contribution. This visualization is invaluable for presenting performance attribution to his investment committee, clearly showing how much of a stock's return is due to broad market movements, specific factor exposures, or genuinely idiosyncratic alpha (skill). The predicted vs. actual plots provide a visual check on the model's fit; a tighter scatter around the 45-degree line indicates a better predictive capability, though for individual stock returns, significant scatter is expected. The final comprehensive report synthesizes all his findings, allowing him to quickly assess each stock's factor fingerprint, performance metrics, and the robustness of the model.

This notebook provides Alex with a powerful and reproducible Python workflow, moving him away from manual spreadsheets and enabling deeper, more dynamic insights into his portfolio's risk and return drivers. This also serves as an interpretable baseline for him to compare more complex machine learning models in the future.
