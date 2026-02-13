id: 698f73b053049d8a753b8dad_documentation
summary: Lab 4: Predicting Stock Beta (Regression) Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Mastering Stock Beta Prediction and Factor Analysis with Streamlit

## 1. Introduction, Overview & Data Setup
Duration: 0:10:00

Welcome to QuLab: Lab 4, where we delve into advanced quantitative finance using Streamlit! This codelab is designed for developers and quantitative analysts looking to build powerful, interactive tools for financial modeling. We will explore how to implement and leverage classical factor models to understand and predict stock behavior.

<aside class="positive">
<b>Why is this application important?</b>
In modern finance, understanding the drivers of asset returns is crucial for portfolio management, risk assessment, and performance attribution. Moving beyond simple market beta (CAPM) to multi-factor models like the Fama-French 3-Factor model provides a more nuanced view, enabling superior investment decisions. This Streamlit application automates a complex analytical workflow, making it reproducible, scalable, and accessible, which is a significant upgrade from error-prone spreadsheet-based analyses. It also serves as an interpretable baseline for comparing against more complex machine learning models.
</aside>

**Persona Introduction:**
Our journey follows **Alex**, a CFA Charterholder and Portfolio Manager at 'Alpha Investments'. Alex is keen to transition his robust financial analysis from traditional tools to Python to achieve greater automation, scalability, and deeper insights into his portfolio's performance and risk factors. His primary goal is to leverage factor models for:
*   **Performance Attribution:** Decomposing returns into market, size, value, and idiosyncratic components.
*   **Systematic Risk Management:** Identifying and managing exposure to common risk factors.
*   **Return Forecasting:** Projecting asset performance under various macroeconomic scenarios.

This codelab will guide you through Alex's Python workflow, covering data acquisition, model estimation, diagnostic testing, dynamic analysis, and scenario planning, all integrated into a user-friendly Streamlit interface.

### Application Architecture Overview

The Streamlit application provides an interactive web interface for performing complex quantitative analyses. At its core, it leverages Python libraries like `yfinance` for data retrieval, `pandas` for data manipulation, `statsmodels` for econometric regressions, and `matplotlib`/`seaborn` for visualizations.

Here's a simplified architectural flow:

```mermaid
graph TD
    A[Streamlit Web App] --> B{User Input: Tickers, Dates, Scenarios};
    B --> C[Data Acquisition & Preparation];
    C --> D[Yahoo Finance (Stock Data)];
    C --> E[Kenneth French Data Library (FF Factors)];
    D & E --> F[Data Merging & Preprocessing (Excess Returns)];
    F --> G[CAPM Baseline];
    F --> H[Fama-French 3-Factor Model];
    G --> I[Regression Results & Metrics];
    H --> I;
    H --> J[Model Diagnostics];
    H --> K[Rolling Betas Calculation];
    H --> L[Scenario Analysis];
    I --> M[Visualizations & Reports];
    J --> M;
    K --> M;
    L --> M;
    M --> A;
```
**Explanation of the Flow:**
1.  **User Input:** The Streamlit app provides interactive widgets (text inputs, date pickers, sliders) for Alex to specify stocks, timeframes, and model parameters.
2.  **Data Acquisition:** The app fetches historical stock prices from Yahoo Finance and Fama-French factor data (Market Excess Return, Small Minus Big, High Minus Low, Risk-Free Rate) from Kenneth French's data library.
3.  **Data Preparation:** The raw data is cleaned, aligned (monthly frequency), and transformed into excess returns (asset return - risk-free rate), ready for regression.
4.  **Model Estimation:**
    *   **CAPM Baseline:** A single-factor regression model relating asset excess returns to market excess returns.
    *   **Fama-French 3-Factor Model:** An extended model incorporating size (SMB) and value (HML) factors.
5.  **Advanced Analysis:**
    *   **Model Diagnostics:** Checks for OLS assumption violations (autocorrelation, heteroskedasticity, multicollinearity).
    *   **Rolling Betas:** Computes factor sensitivities over a moving window to capture their dynamic nature.
    *   **Scenario Analysis:** Projects future returns based on estimated factor exposures and hypothetical factor returns.
6.  **Visualizations & Reports:** All results are presented through interactive charts, summary tables, and clear interpretations within the Streamlit interface.

### Data Acquisition and Preparation

The first step is to acquire and prepare the necessary financial data. This involves:
*   Fetching historical monthly total returns for selected stocks.
*   Retrieving the Fama-French 3-Factor data (Mkt-RF, SMB, HML) and the Risk-Free Rate (RF).
*   Merging these datasets and calculating excess returns for the stocks.

**Interacting with the App:**
Navigate to the "Introduction & Data Setup" page in the sidebar.

1.  **Enter Stock Tickers:** In the text input field, provide comma-separated stock tickers (e.g., `AAPL, BRK-B, TSLA, JNJ`). The application initializes with a default set of tickers.
2.  **Select Date Range:** Use the `Start Date` and `End Date` pickers to define your analysis period.
3.  **Retrieve Data:** Click the **"Retrieve and Prepare Data"** button. The application will fetch the data, merge it, and display the head of the merged DataFrame. It will also generate a plot comparing the first selected stock's excess return with the market excess return (`Mkt-RF`) to visually verify data alignment.

**Under the Hood (`source.py` - `retrieve_and_merge_data` function):**
This function (not provided directly in the Streamlit code, but imported from `source.py`) handles the data fetching and cleaning.
```python
# Conceptual example of retrieve_and_merge_data function logic
# (Actual implementation is in source.py)
import yfinance as yf
import pandas_datareader.data as web

def retrieve_and_merge_data(tickers, start_date_str, end_date_str):
    # 1. Fetch Fama-French 3-Factor data
    ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench',
                             start=start_date_str, end=end_date_str)[0]
    # Convert daily to monthly, ensuring correct end-of-month alignment
    # ... (complex logic for monthly conversion and cleaning)
    ff_data = ff_data.resample('M').last() # Example, actual logic is more robust
    ff_data = ff_data.apply(pd.to_numeric, errors='coerce') / 100 # Convert percent to decimal

    # 2. Fetch stock data from Yahoo Finance
    stock_dfs = []
    for ticker in tickers:
        stock_df = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)
        # Calculate monthly returns (adjusted close)
        stock_df['Monthly_Return'] = stock_df['Adj Close'].pct_change()
        stock_df = stock_df.resample('M').last() # Ensure monthly frequency
        stock_df = stock_df[['Monthly_Return']].dropna()
        stock_df = stock_df.rename(columns={'Monthly_Return': f'{ticker}_Return'})
        stock_dfs.append(stock_df)

    # 3. Merge stock data with Fama-French data
    df_merged = ff_data.copy()
    for stock_df in stock_dfs:
        df_merged = df_merged.merge(stock_df, left_index=True, right_index=True, how='inner')

    # 4. Calculate Excess Returns
    for ticker in tickers:
        df_merged[f'{ticker}_excess'] = df_merged[f'{ticker}_Return'] - df_merged['RF']

    return df_merged
```

<aside class="negative">
<b>Practitioner Warning:</b> Fama-French factors typically use end-of-month dates while Yahoo Finance may have slightly different conventions (e.g., last trading day). Always verify that the merge is correct by spot-checking known dates (e.g., March 2020 COVID crash should show large negative Mkt-RF). A one-month misalignment would produce meaningless regressions. Also note that Fama-French returns are often in percent (e.g., 2.5 = 2.5%) while Yahoo Finance returns are in decimal (0.025). Convert before merging. The provided `source.py` is assumed to handle these conversions correctly.
</aside>

After successful data retrieval, Alex confirms the data is aligned and ready for model estimation.

## 2. CAPM Baseline: Understanding Market Sensitivity
Duration: 0:08:00

With the data prepared, Alex establishes a baseline understanding of each stock's sensitivity to the overall market using the Capital Asset Pricing Model (CAPM). This single-factor model explains the expected return of an asset based on its market risk.

The CAPM is represented by the following regression equation:
$$ R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \epsilon_{i,t} $$
Where:
*   $R_{i,t} - R_{f,t}$ is the **excess return** of asset $i$ at time $t$.
*   $\alpha_i$ (Jensen's Alpha) is the asset's **abnormal return** not explained by the market factor. A positive and statistically significant $\alpha_i$ indicates outperformance after adjusting for market risk.
*   $\beta_{i,M}$ (Market Beta) measures the asset's **sensitivity to market movements**. A $\beta_{i,M} > 1$ implies higher market sensitivity than the average stock, while $\beta_{i,M} < 1$ implies lower sensitivity.
*   $R_{m,t} - R_{f,t}$ is the **market excess return** at time $t$.
*   $\epsilon_{i,t}$ is the **idiosyncratic error term**.

Alex will perform this regression for each of his target stocks to obtain their individual market betas and Jensen's alpha, along with statistical significance.

**Interacting with the App:**
Navigate to the "CAPM Baseline" page.

1.  **Run CAPM:** Click the **"Run CAPM Regression for All Stocks"** button.
2.  **Review Results:** For each selected stock, the app will display the `statsmodels` regression summary and key interpretations:
    *   Annualized Alpha and its p-value.
    *   Market Beta ($\beta_M$) and its p-value.
    *   R-squared value.
    *   Information Ratio.

**Under the Hood (`source.py` - `run_capm_regression` function):**
This function performs an OLS regression using `statsmodels` for the CAPM.
```python
# Conceptual example of run_capm_regression function logic
# (Actual implementation is in source.py)
import statsmodels.api as sm

def run_capm_regression(df, stock_ticker):
    y = df[f'{stock_ticker}_excess']
    X = sm.add_constant(df['Mkt_RF']) # Add constant for alpha
    
    model = sm.OLS(y, X).fit()
    
    # Extract key metrics
    alpha = model.params['const']
    beta_M = model.params['Mkt_RF']
    
    # Annualize alpha and calculate information ratio
    alpha_ann = alpha * 12
    std_error_epsilon = model.resid.std()
    information_ratio = alpha_ann / (std_error_epsilon * np.sqrt(12))

    return {
        'model': model,
        'alpha_ann': alpha_ann,
        'alpha_pval': model.pvalues['const'],
        'beta_M': beta_M,
        'beta_M_pval': model.pvalues['Mkt_RF'],
        'r_squared': model.rsquared,
        'information_ratio': information_ratio
    }
```
Alex reviews the outputs. He notes the market beta ($\beta_M$) values and pays close attention to Jensen's Alpha ($\alpha$) and its p-value. A high p-value for alpha (e.g., > 0.05) indicates that the stock's abnormal return is not statistically significant and could be due to random chance. The R-squared value tells him the proportion of the stock's excess return variance explained by the market factor.

## 3. Fama-French 3-Factor Model: Deeper Insights
Duration: 0:12:00

While CAPM provides a basic understanding, Alex knows that investment performance is often driven by more than just market risk. He moves to the Fama-French 3-Factor Model, which adds size (SMB) and value (HML) factors, offering a richer explanation of asset returns and a more nuanced performance attribution.

The Fama-French 3-Factor Model is given by:
$$ R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \beta_{i,S}SMB_t + \beta_{i,H}HML_t + \epsilon_{i,t} $$
Where:
*   $\beta_{i,S}$ (Size Beta) measures the asset's exposure to the small-cap factor. A positive $\beta_{i,S}$ suggests a tilt towards smaller companies.
*   $\beta_{i,H}$ (Value Beta) measures the asset's exposure to the value factor. A positive $\beta_{i,H}$ suggests a tilt towards value stocks (high book-to-market), while a negative $\beta_{i,H}$ indicates a growth stock tilt (low book-to-market).
*   Other terms ($R_{i,t} - R_{f,t}$, $\alpha_i$, $\beta_{i,M}$, $R_{m,t} - R_{f,t}$, $\epsilon_{i,t}$) are as defined in the CAPM.

Alex will run this model for all stocks, compare their factor exposures (their "factor fingerprints"), and quantify the incremental explanatory power of the additional factors (SMB and HML). He'll also use the Information Ratio, defined as:
$$ IR = \frac{\hat{\alpha}_{\text{ann}}}{\hat{\sigma}_{\epsilon}\sqrt{12}} = \frac{\text{Annualized Alpha}}{\text{Annualized Tracking Error}} $$
The Information Ratio measures the risk-adjusted abnormal return, where $|IR| > 0.5$ is often considered strong performance.

**Interacting with the App:**
Navigate to the "Fama-French 3-Factor Model" page.

1.  **Run FF3:** Click the **"Run Fama-French 3-Factor Regression for All Stocks"** button.
2.  **Review Results:** For each stock, the `statsmodels` summary will be displayed, showing coefficients for Mkt-RF, SMB, and HML, along with the constant ($\alpha$).

**Under the Hood (`source.py` - `run_ff3_regression` function):**
This function extends the CAPM regression to include SMB and HML factors.
```python
# Conceptual example of run_ff3_regression function logic
# (Actual implementation is in source.py)
import statsmodels.api as sm

def run_ff3_regression(df, stock_ticker):
    y = df[f'{stock_ticker}_excess']
    X = sm.add_constant(df[['Mkt_RF', 'SMB', 'HML']]) # Add factors
    
    model = sm.OLS(y, X).fit()
    
    # Extract key metrics
    alpha = model.params['const']
    beta_M = model.params['Mkt_RF']
    beta_S = model.params['SMB']
    beta_H = model.params['HML']

    # Annualize alpha and calculate information ratio
    alpha_ann = alpha * 12
    std_error_epsilon = model.resid.std()
    information_ratio = alpha_ann / (std_error_epsilon * np.sqrt(12))

    return {
        'model': model,
        'alpha_ann': alpha_ann,
        'alpha_pval': model.pvalues['const'],
        'beta_M': beta_M,
        'beta_S': beta_S,
        'beta_H': beta_H,
        'r_squared': model.rsquared,
        'adj_r_squared': model.adj_rsquared,
        'information_ratio': information_ratio
    }
```

After running the regressions, a **Comparative Factor Exposure & Performance Table** is displayed, summarizing key metrics for both CAPM and FF3 models, including the improvement in R-squared.

Alex examines the outputs and the comparative table. He notes how stocks exhibit distinct factor fingerprints, like `TSLA`'s high $\beta_M$ and negative $\beta_H$ (growth tilt), versus `BRK-B`'s more moderate $\beta_M$ and positive $\beta_H$ (value tilt). The generally higher R-squared values for the FF3 model indicate improved explanatory power.

**Visualizations:**
The page also presents two important charts:
*   **Factor Beta Comparison Bar Chart:** A bar chart comparing the Mkt, SMB, and HML betas across all selected stocks, providing a visual "factor fingerprint" for each.
*   **Security Market Line (SML) Plot:** This plot shows the annualized average excess returns of the stocks against their Fama-French market betas. It helps visualize how stocks deviate from the theoretical SML, offering insights into their risk-adjusted performance.

## 4. Model Diagnostics: Ensuring Robustness
Duration: 0:15:00

Before relying on the factor model for critical investment decisions, Alex must perform diagnostic tests to check if the underlying assumptions of Ordinary Least Squares (OLS) regression are met. Violations of these assumptions (e.g., autocorrelation, heteroskedasticity, multicollinearity) can lead to inefficient or biased parameter estimates and incorrect statistical inferences (e.g., t-statistics, p-values). This step ensures the robustness of his analysis.

He will check for:

*   **Autocorrelation (Durbin-Watson statistic):** Checks if residuals are correlated over time.
    $$ DW \approx 2(1 - \rho_1) $$
    Where $\rho_1$ is the first-order autocorrelation of residuals. A value close to 2 indicates no autocorrelation. For financial time series, positive autocorrelation ($DW < 2$, especially $< 1.5$) can indicate momentum effects or missing factors.

*   **Heteroskedasticity (Breusch-Pagan test):** Checks if the variance of the residuals is constant across all levels of independent variables. Heteroskedasticity (Breusch-Pagan p-value $< 0.05$) leads to inefficient estimates and incorrect standard errors.

*   **Multicollinearity (Variance Inflation Factor - VIF):** Checks if independent variables are highly correlated with each other.
    $$ VIF_j = \frac{1}{1 - R_j^2} $$
    Where $R_j^2$ is the R-squared from regressing factor $j$ on all other factors. High multicollinearity ($VIF > 5$ or $10$) can make coefficient estimates unstable and difficult to interpret.

<aside class="negative">
<b>Practitioner Warning:</b> Heteroskedasticity is common in financial data. If the Breusch-Pagan test rejects homoskedasticity (typical for equity returns, where volatility clusters in crises), switch to Newey-West HAC standard errors for robust inference. HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors produce valid t-statistics and p-values even when classical OLS assumptions are violated—a critical technique for financial time-series.
</aside>

**Interacting with the App:**
Navigate to the "Model Diagnostics" page.

1.  **Select Stock:** Choose a stock from the dropdown menu for which you want to run diagnostics.
2.  **Run Diagnostics:** Click the **"Run Diagnostics"** button.
3.  **Review Results:** The app will display the Durbin-Watson statistic, Breusch-Pagan p-value, and VIF results for each factor, along with their interpretations.
4.  **Analyze Residual Plots:** A **Diagnostic 4-Panel Plot of Residuals** will be displayed, including:
    *   Residuals Over Time: To visually check for patterns or trends.
    *   Residuals vs Fitted Values: To check for heteroskedasticity (fanning out or clustering).
    *   Q-Q Plot of Residuals: To assess normality of residuals.
    *   Histogram of Residuals: To visualize the distribution of residuals.

**Under the Hood (`source.py` - `run_diagnostic_tests` function):**
This function utilizes `statsmodels` for the diagnostic tests.
```python
# Conceptual example of run_diagnostic_tests function logic
# (Actual implementation is in source.py)
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run_diagnostic_tests(model, X_for_vif, stock_ticker):
    results = {}
    
    # Durbin-Watson
    dw_stat = sms.durbin_watson(model.resid)
    results['dw_stat'] = dw_stat
    if dw_stat < 1.5 or dw_stat > 2.5:
        results['dw_interpretation'] = "Potential positive or negative autocorrelation."
    else:
        results['dw_interpretation'] = "No significant autocorrelation detected."

    # Breusch-Pagan test for heteroskedasticity
    # model.model.exog are the independent variables (factors)
    bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
    results['bp_pvalue'] = bp_test[1] # p-value
    if bp_test[1] < 0.05:
        results['bp_interpretation'] = "Heteroskedasticity detected (p < 0.05)."
    else:
        results['bp_interpretation'] = "No significant heteroskedasticity detected."

    # VIF for multicollinearity
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_for_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_for_vif.values, i) 
                       for i in range(len(X_for_vif.columns))]
    results['vif_results'] = vif_data.set_index('feature')['VIF'].to_dict()
    
    if any(v > 5 for v in results['vif_results'].values()):
        results['vif_interpretation'] = "Potential multicollinearity detected (some VIF > 5)."
    else:
        results['vif_interpretation'] = "No significant multicollinearity detected."

    return results
```
Alex reviews the diagnostic test results and the 4-panel plots. He notes that while Durbin-Watson and VIFs are generally acceptable for Fama-French factors, heteroskedasticity (Breusch-Pagan p-value < 0.05) is often present. This highlights the need for **HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors** in real-world reporting to ensure valid statistical inferences despite these common financial data characteristics.

## 5. Rolling Betas: Capturing Dynamic Risk
Duration: 0:10:00

Static, full-sample betas can mask how a stock's sensitivity to factors changes over time, especially during different market regimes or significant economic events. As a Portfolio Manager, Alex needs to understand this dynamic nature for effective risk management and tactical asset allocation. He will compute and visualize rolling betas over a defined window (e.g., 36 months) to observe how these exposures evolve.

This technique involves running the factor regression repeatedly on a moving window of historical data. The resulting time series of betas provides insights into how the stock's "factor fingerprint" adapts to changing market conditions.

**Interacting with the App:**
Navigate to the "Rolling Betas" page.

1.  **Select Stock:** Choose a stock from the dropdown menu for rolling beta analysis.
2.  **Adjust Rolling Window Size:** Use the slider to select the desired window size (in months), e.g., 36 months.
3.  **Calculate & Plot:** Click the **"Calculate & Plot Rolling Betas"** button.
4.  **Review Plot:** A time-series plot of the rolling betas (Mkt, SMB, HML) will be displayed. Significant market events (e.g., COVID-19 Crash, Inflation/Rate Hikes) are annotated to help interpret beta changes.

**Under the Hood (`source.py` - `calculate_rolling_betas` function):**
This function iterates through the dataset, applying the Fama-French 3-Factor regression on a rolling window.
```python
# Conceptual example of calculate_rolling_betas function logic
# (Actual implementation is in source.py)
import statsmodels.api as sm

def calculate_rolling_betas(df, stock_ticker, window_size):
    rolling_betas = []
    
    y_col = f'{stock_ticker}_excess'
    X_cols = ['Mkt_RF', 'SMB', 'HML']

    for i in range(len(df) - window_size + 1):
        window_df = df.iloc[i : i + window_size]
        
        y = window_df[y_col]
        X = sm.add_constant(window_df[X_cols])
        
        if len(y) > len(X_cols) + 1: # Ensure enough data points for regression
            model = sm.OLS(y, X).fit()
            betas = model.params.to_dict()
            betas['Date'] = window_df.index[-1] # End date of the window
            rolling_betas.append(betas)

    rolling_betas_df = pd.DataFrame(rolling_betas).set_index('Date')
    return rolling_betas_df[['Mkt_RF', 'SMB', 'HML']] # Return Betas
```
Alex analyzes the rolling beta plots. He observes how `TSLA`'s market beta ($\beta_M$) might increase during periods of market stress, like the COVID-19 crash, indicating it becomes more sensitive to market movements during downturns. This dynamic view of factor exposures is critical for understanding the time-varying risk profile of his portfolio holdings.

## 6. Scenario Analysis: Forward-Looking Projections
Duration: 0:10:00

One of the most powerful applications of factor models for Alex is to project expected returns under various hypothetical macroeconomic scenarios. This shifts the analysis from purely backward-looking performance attribution to a forward-looking risk management and strategic planning tool.

By defining reasonable expected returns for the Fama-French factors in different economic environments, Alex can estimate how his target stocks might perform.

The scenario projection uses the estimated betas from the Fama-French 3-factor model:
$$ E[R_i - R_f] = \hat{\alpha}_i + \hat{\beta}_{i,M} E[R_m - R_f] + \hat{\beta}_{i,S} E[SMB] + \hat{\beta}_{i,H} E[HML] $$
Where $E[...]$ denotes the expected value of the factors under a specific scenario, and $\hat{\alpha}$, $\hat{\beta}$ are the estimated coefficients from the full-sample regression.

Alex will define several plausible scenarios and then calculate the projected annualized excess return for each stock.

**Interacting with the App:**
Navigate to the "Scenario Analysis" page.

1.  **Define Macroeconomic Scenarios:** The page pre-populates with several common scenarios (e.g., Base Case, Market Crash, Value Rotation). For each scenario, you can adjust the expected monthly returns for Mkt-RF, SMB, and HML. You can also add new scenarios or delete existing ones.
2.  **Project Returns:** Click the **"Project Returns Under Scenarios"** button.
3.  **Review Projections:** A summary table of projected annualized excess returns for all stocks under each defined scenario will be displayed.

**Under the Hood (`source.py` - `project_returns_under_scenarios` function):**
This function applies the factor model equation using the estimated betas and the user-defined scenario factor returns.
```python
# Conceptual example of project_returns_under_scenarios function logic
# (Actual implementation is in source.py)
import pandas as pd

def project_returns_under_scenarios(ff3_model_params, macro_scenarios):
    projections_list = []
    
    alpha = ff3_model_params['const']
    beta_M = ff3_model_params['Mkt_RF']
    beta_S = ff3_model_params['SMB']
    beta_H = ff3_model_params['HML']

    for scenario_name, factor_returns in macro_scenarios.items():
        E_Mkt_RF = factor_returns['Mkt_RF']
        E_SMB = factor_returns['SMB']
        E_HML = factor_returns['HML']
        
        # Project monthly excess return
        projected_monthly_excess_return = alpha + \
                                          beta_M * E_Mkt_RF + \
                                          beta_S * E_SMB + \
                                          beta_H * E_HML
        
        # Annualize
        projected_annual_excess_return = projected_monthly_excess_return * 12
        
        projections_list.append({
            'Scenario': scenario_name,
            'Projected_Monthly_Excess_Return': projected_monthly_excess_return,
            'Projected_Annual_Excess_Return': projected_annual_excess_return
        })
    
    return pd.DataFrame(projections_list)
```
Alex reviews the projected returns under different scenarios. He observes that in a 'Market Crash' scenario, `TSLA` (high market beta) shows a significantly larger projected negative return compared to `JNJ` (lower market beta). In a 'Value Rotation' scenario, `BRK-B` (positive value beta) might be projected to perform relatively better than `AAPL` or `TSLA` (negative value/growth tilt). This table provides Alex with critical insights for stress testing his portfolio, adjusting his risk exposure to specific factors, and informing his discussions with the investment committee.

## 7. Performance Attribution & Report
Duration: 0:15:00

To complete his comprehensive analysis, Alex wants to visualize the contribution of each factor to the cumulative excess return of a stock. This "cumulative return decomposition" helps him attribute performance to market, size, and value factors versus the stock's idiosyncratic alpha. He also wants a visual comparison of the model's predicted versus actual returns and a final summary of all key metrics for easy reporting.

The cumulative contribution of each factor at time $T$ is given by:
$$ \text{Cumulative Factor Contribution}_X = \sum_{t=1}^T \hat{\beta}_{X} \cdot F_{X,t} $$
Where $F_{X,t}$ is the factor return for factor $X$ at time $t$. The cumulative alpha contribution is $\sum_{t=1}^T \hat{\alpha}$.

**Interacting with the App:**
Navigate to the "Performance Attribution & Report" page.

1.  **Select Stock:** Choose a stock from the dropdown menu for performance attribution.
2.  **Generate Attribution:** Click the **"Generate Performance Attribution"** button.
3.  **Review Charts:**
    *   **Cumulative Return Decomposition Chart:** This stacked area chart visually breaks down the cumulative actual excess return into contributions from the Market Factor, SMB Factor, HML Factor, Alpha, and Residual (unexplained) components.
    *   **Predicted vs. Actual Scatter Plot:** This scatter plot shows how well the Fama-French 3-Factor model's predicted returns align with the actual excess returns for the selected stock. A 45-degree line indicates perfect prediction.
4.  **Comprehensive Report:** A final, formatted table summarizing all key findings (FF3 betas, alpha, R-squared, IR, diagnostic results) for all stocks is displayed.

**Under the Hood (Calculation for Cumulative Return Decomposition):**
The app calculates the contribution of each component over time.
```python
# Conceptual calculation for Cumulative Return Decomposition
# (Logic implemented directly in Streamlit app based on model results)
# Assuming ff3_model_params, y (actual excess return), X (factors) are available

alpha_term = ff3_model_params['const']
beta_M_term = ff3_model_params['Mkt_RF']
beta_S_term = ff3_model_params['SMB']
beta_H_term = ff3_model_params['HML']

# Monthly contributions
market_contribution_monthly = beta_M_term * X['Mkt_RF']
smb_contribution_monthly = beta_S_term * X['SMB']
hml_contribution_monthly = beta_H_term * X['HML']
alpha_contribution_monthly = pd.Series(alpha_term, index=X.index)

# Model's total predicted return
total_model_return_monthly = market_contribution_monthly + smb_contribution_monthly + \
                             hml_contribution_monthly + alpha_contribution_monthly

# Residual is the unexplained portion
epsilon_contribution_monthly = y - total_model_return_monthly

# Cumulative sums for plotting
cumulative_market = market_contribution_monthly.cumsum()
cumulative_smb = smb_contribution_monthly.cumsum()
cumulative_hml = hml_contribution_monthly.cumsum()
cumulative_alpha = alpha_contribution_monthly.cumsum()
cumulative_residual = epsilon_contribution_monthly.cumsum()
cumulative_actual_excess_return = y.cumsum()
```
Alex examines the cumulative return decomposition plots. This visualization is invaluable for presenting performance attribution to his investment committee, clearly showing how much of a stock's return is due to broad market movements, specific factor exposures, or genuinely idiosyncratic alpha (skill). The predicted vs. actual plots provide a visual check on the model's fit. The final comprehensive report synthesizes all his findings, allowing him to quickly assess each stock's factor fingerprint, performance metrics, and the robustness of the model.

This workflow provides Alex with a powerful and reproducible Python workflow, moving him away from manual spreadsheets and enabling deeper, more dynamic insights into his portfolio's risk and return drivers. This also serves as an interpretable baseline for him to compare more complex machine learning models in the future.
