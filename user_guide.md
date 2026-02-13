id: 698f73b053049d8a753b8dad_user_guide
summary: Lab 4: Predicting Stock Beta (Regression) User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Predicting Stock Beta and Factor Exposures

## 1. Introduction & Data Setup
Duration: 0:08:00

Welcome to QuLab's Lab 4, an interactive guide designed for financial professionals to delve into the world of factor models using a Streamlit application. This codelab will empower you, the user, to transition from traditional spreadsheet-based analysis to a more robust, scalable, and reproducible Python-powered workflow for financial analysis.

<aside class="positive">
<b>Persona:</b> Meet Alex, a CFA Charterholder and Portfolio Manager at 'Alpha Investments'. Alex's primary objective is to move beyond simple market beta to understand the deeper drivers of investment performance through multi-factor models. He aims to enhance his capabilities in performance attribution, systematic risk management, and return forecasting. This application streamlines these complex tasks, making them accessible and efficient.
</aside>

### Understanding the Importance
In investment management, understanding how asset returns relate to various risk factors is crucial. The Capital Asset Pricing Model (CAPM) introduced the concept of market risk (beta). However, subsequent research, notably the Fama-French 3-Factor Model, demonstrated that other factors like 'size' (SMB) and 'value' (HML) also systematically influence returns. This application helps Alex dissect these influences.

### Data Acquisition and Preparation
The first step in any robust financial analysis is gathering and preparing reliable data. This application automates the process of acquiring historical monthly total returns for your chosen stocks, along with the widely-used Fama-French factor returns and the risk-free rate.

<aside class="negative">
<b>Practitioner Warning:</b> Data alignment is paramount! Fama-French factors often use end-of-month dates, while stock data from sources like Yahoo Finance might follow different conventions. A one-month misalignment can render your regressions meaningless. Always visually inspect your data, and ensure factor returns (often in percentage, e.g., 2.5%) are converted to decimals (0.025) before merging with stock returns.
</aside>

#### Your Turn: Configure Data Retrieval
1.  **Enter Stock Tickers:** In the "Enter stock tickers" box, you'll see a default list (e.g., `AAPL, BRK-B, TSLA, JNJ`). You can modify this by adding or removing tickers, ensuring they are comma-separated.
2.  **Select Date Range:** Choose a `Start Date` and `End Date` for your analysis period. The application retrieves monthly data.
3.  **Retrieve Data:** Click the **"Retrieve and Prepare Data"** button. The application will fetch the necessary data and display a preview of the merged DataFrame's head, showing your stock's excess returns alongside the Fama-French factors and risk-free rate.
4.  **Verify Data Alignment:** After retrieval, the application will display a plot comparing one of your selected stock's excess returns with the Market Excess Return (Mkt-RF). This visual check is essential to confirm that the data is aligned and appears reasonable. For instance, observe if periods of general market downturns (like the COVID-19 crash in March 2020) show corresponding drops in both market and stock returns.

<aside class="console">
Example of Data Retrieval & Preview:
Enter stock tickers (comma-separated): AAPL, BRK-B, TSLA, JNJ
Start Date: 2014-01-01
End Date: 2024-01-01
Retrieve and Prepare Data (button clicked)
</aside>

After successful retrieval, Alex confirms the data is aligned and ready. He notes the dataset contains the specified number of months of data, forming the foundation for all subsequent analyses.

## 2. CAPM Baseline
Duration: 0:07:00

### Establishing a Baseline: The Capital Asset Pricing Model (CAPM)
Before exploring more complex models, Alex begins with the fundamental Capital Asset Pricing Model (CAPM). This single-factor model provides a baseline understanding of each stock's sensitivity to overall market movements, known as **market risk**.

The CAPM is expressed as:
$$ R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \epsilon_{i,t} $$

Let's break down the components:
*   $R_{i,t} - R_{f,t}$: This is the **excess return** of asset $i$ at time $t$. It's the asset's return minus the risk-free rate, representing the return earned for taking on risk.
*   $\alpha_i$ (Jensen's Alpha): Represents the asset's **abnormal return** not explained by the market factor. A positive and statistically significant $\alpha_i$ indicates that the asset has outperformed the market given its risk, implying a manager's skill.
*   $\beta_{i,M}$ (Market Beta): This is a crucial measure of the asset's **sensitivity to market movements**.
    *   A $\beta_{i,M} > 1$ implies the stock is more volatile than the market (e.g., if the market moves 1%, the stock moves more than 1%).
    *   A $\beta_{i,M} < 1$ implies lower volatility than the market (e.g., if the market moves 1%, the stock moves less than 1%).
    *   A $\beta_{i,M} = 1$ means the stock moves in tandem with the market.
*   $R_{m,t} - R_{f,t}$: This is the **market excess return** at time $t$. It's the market's return minus the risk-free rate, representing the market risk premium.
*   $\epsilon_{i,t}$: This is the **idiosyncratic error term**, representing the portion of the asset's return that is not explained by the market factor.

Alex will perform this regression for each of his target stocks to obtain their individual market betas and Jensen's alpha, along with statistical significance.

#### Your Turn: Run CAPM Regression
1.  Navigate to the "CAPM Baseline" page using the sidebar.
2.  Ensure you have retrieved data on the "Introduction & Data Setup" page. If not, a warning will appear.
3.  Click the **"Run CAPM Regression for All Stocks"** button. The application will compute the CAPM for each of your selected tickers.
4.  **Review the Output:** For each stock, you will see a detailed regression summary. Pay attention to:
    *   **Annualized Alpha:** Presented as a percentage, this indicates the annual abnormal return. Its p-value tells you if this alpha is statistically significant (typically, a p-value < 0.05 is considered significant).
    *   **Market Beta ($\beta_M$):** This value quantifies the stock's market sensitivity. Its p-value indicates statistical significance.
    *   **R-squared:** This value (between 0 and 1) represents the proportion of the stock's excess return variance that is explained by the market factor. A higher R-squared means the market factor is a better predictor of the stock's returns.
    *   **Information Ratio (IR):** A measure of risk-adjusted return.

<aside class="console">
Example of CAPM output (partial summary):
CAPM Results for AAPL
                              OLS Regression Results
====================================================================================
Dep. Variable:         AAPL_excess   R-squared:                       0.354
Model:                         OLS   Adj. R-squared:                  0.353
Method:              Least Squares   F-statistic:                     652.8
Date:             Wed, 24 Jul 2024   Prob (F-statistic):           1.41e-114
Time:                     10:00:00   Log-Likelihood:                 1376.6
No. Observations:              120   AIC:                            -2749.
Df Residuals:                  118   BIC:                            -2744.
No. of Coefficients:             2
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]

const          0.0094      0.002      4.053      0.000       0.005       0.014
Mkt_RF         1.0927      0.043     25.550      0.000       1.008       1.177
==============================================================================
Interpretation for AAPL:
  - Annualized Alpha: 11.28% (p-value: 0.0001)
  - Market Beta (𝛽𝑀): 1.093 (p-value: 0.0000)
  - R-squared: 0.354
  - Information Ratio: 0.814
</aside>

Alex reviews the outputs. He notes the market beta values and pays close attention to Jensen's Alpha ($\alpha$) and its p-value. A high p-value for alpha (e.g., > 0.05) indicates that the stock's abnormal return is not statistically significant and could be due to random chance. The R-squared value tells him the proportion of the stock's excess return variance explained by the market factor.

## 3. Fama-French 3-Factor Model
Duration: 0:10:00

### Deeper Insights with Fama-French 3-Factor Model
While CAPM offers a foundational understanding, Alex knows that investment performance is often influenced by more than just market risk. He now moves to the Fama-French 3-Factor Model (FF3), which expands on CAPM by adding two additional factors: **size (SMB)** and **value (HML)**. This model provides a richer explanation of asset returns and allows for a more nuanced performance attribution.

The Fama-French 3-Factor Model is given by:
$$ R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \beta_{i,S}SMB_t + \beta_{i,H}HML_t + \epsilon_{i,t} $$

Here's what the new terms mean:
*   $\beta_{i,S}$ (Size Beta): Measures the asset's exposure to the **small-cap factor (SMB - Small Minus Big)**. A positive $\beta_{i,S}$ suggests that the stock tends to perform better when small-cap stocks outperform large-cap stocks, indicating a tilt towards smaller companies.
*   $\beta_{i,H}$ (Value Beta): Measures the asset's exposure to the **value factor (HML - High Minus Low)**. A positive $\beta_{i,H}$ suggests a tilt towards "value" stocks (companies with high book-to-market ratios, often seen as undervalued). A negative $\beta_{i,H}$ indicates a tilt towards "growth" stocks (companies with low book-to-market ratios, often associated with higher growth potential).
*   Other terms ($\alpha_i$, $\beta_{i,M}$, $R_{m,t} - R_{f,t}$, $\epsilon_{i,t}$) are defined as in the CAPM.

Alex will run this model for all stocks, compare their factor exposures (their "factor fingerprints"), and quantify the incremental explanatory power of the additional factors (SMB and HML). He'll also use the **Information Ratio (IR)**, which measures the risk-adjusted abnormal return:
$$ IR = \frac{\hat{\alpha}_{\text{ann}}}{\hat{\sigma}_{\epsilon}\sqrt{12}} = \frac{\text{Annualized Alpha}}{\text{Annualized Tracking Error}} $$
An $|IR| > 0.5$ is generally considered strong performance, indicating a consistent ability to generate alpha relative to the volatility of that alpha.

#### Your Turn: Run Fama-French 3-Factor Regression
1.  Navigate to the "Fama-French 3-Factor Model" page.
2.  Ensure you have retrieved data.
3.  Click the **"Run Fama-French 3-Factor Regression for All Stocks"** button. The application will perform the FF3 regression for each stock.
4.  **Review the Output:** Similar to CAPM, a detailed regression summary will be displayed, now including coefficients and p-values for SMB and HML.

#### Comparative Factor Exposure & Performance Analysis
After running the FF3 regressions, the application provides a comparative table summarizing key metrics from both CAPM and FF3 for all your stocks.

<aside class="console">
Example of Comparative Table (partial):
Stock  CAPM_Alpha_Ann CAPM_Alpha_pvalue CAPM_Beta_M CAPM_R_squared CAPM_IR FF3_Alpha_Ann FF3_Alpha_pvalue FF3_Beta_M FF3_Beta_S FF3_Beta_H FF3_R_squared FF3_Adj_R_squared FF3_IR R_squared_Improvement
AAPL      11.28%             0.000       1.093          0.354    0.814          7.18%            0.005      1.050      0.297     -0.298         0.387             0.370 0.536                 0.033
</aside>

Alex examines the outputs and the comparative table. He notes how stocks exhibit distinct factor fingerprints, like `TSLA`'s high $\beta_M$ and negative $\beta_H$ (growth tilt), versus `BRK-B`'s more moderate $\beta_M$ and positive $\beta_H$ (value tilt). The generally higher R-squared values for the FF3 model indicate improved explanatory power compared to CAPM.

#### Visualizing Factor Betas and the Security Market Line (SML)
The application also generates two important plots:

*   **Factor Beta Comparison Bar Chart:** This chart visually compares the Market, Size (SMB), and Value (HML) betas for each stock. This allows Alex to quickly identify which stocks have stronger exposures to different factors.
*   **Security Market Line (SML) Plot:** This plot shows each stock's annualized average excess return against its Market Beta ($\beta_M$). The theoretical SML (red dashed line) represents the expected return for a given market beta based on the market risk premium. Stocks plotting above the SML are potentially outperforming, while those below are underperforming, after accounting for market risk.

## 4. Model Diagnostics
Duration: 0:08:00

### Validating Model Assumptions: Diagnostic Tests
Before relying on any factor model for critical investment decisions, Alex understands the necessity of performing diagnostic tests. These tests check if the underlying assumptions of Ordinary Least Squares (OLS) regression (the method used to estimate the factor models) are met. Violations of these assumptions can lead to inefficient or biased parameter estimates and incorrect statistical inferences (e.g., t-statistics, p-values), which could mislead investment decisions. This step ensures the robustness of his analysis.

Alex will check for:

*   **Autocorrelation (Durbin-Watson statistic):** This test checks if the residuals (the unexplained part of the stock's return) are correlated over time.
    $$ DW \approx 2(1 - \rho_1) $$
    where $\rho_1$ is the first-order autocorrelation of residuals. A Durbin-Watson statistic close to 2 (typically between 1.5 and 2.5) indicates no significant autocorrelation. For financial time series, positive autocorrelation ($DW < 2$, especially $< 1.5$) can indicate momentum effects or missing factors.
*   **Heteroskedasticity (Breusch-Pagan test):** This test checks if the variance of the residuals is constant across all levels of independent variables. **Heteroskedasticity** (indicated by a Breusch-Pagan p-value $< 0.05$) means the error variance is not constant. This does not bias the coefficient estimates but makes them inefficient and leads to incorrect standard errors, making hypothesis tests (like p-values for betas) unreliable.
*   **Multicollinearity (Variance Inflation Factor - VIF):** This test checks if the independent variables (the market, SMB, and HML factors) are highly correlated with each other.
    $$ VIF_j = \frac{1}{1 - R_j^2} $$
    where $R_j^2$ is the R-squared from regressing factor $j$ on all other factors. High multicollinearity ($VIF > 5$ or $10$) can make coefficient estimates unstable and difficult to interpret, as it's hard to isolate the individual effect of each correlated factor.

<aside class="negative">
<b>Practitioner Warning:</b> Heteroskedasticity is very common in financial data, especially for equity returns where volatility tends to cluster during crises. If the Breusch-Pagan test rejects homoskedasticity (which it often will for equity returns), it's crucial to switch to **Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors** for robust statistical inference. HAC standard errors produce valid t-statistics and p-values even when classical OLS assumptions of homoskedasticity and no autocorrelation are violated—a critical technique for financial time-series analysis.
</aside>

#### Your Turn: Run Diagnostic Tests
1.  Navigate to the "Model Diagnostics" page.
2.  Ensure you have run the Fama-French 3-Factor regressions.
3.  **Select a Stock:** Use the dropdown menu to choose one of your stocks for diagnostic testing.
4.  Click the **"Run Diagnostics for [Selected Stock]"** button.

#### Reviewing Diagnostic Results
The application will display the results of the Durbin-Watson, Breusch-Pagan, and VIF tests, along with an interpretation for each.
It also generates a **Diagnostic 4-Panel Plot of Residuals**:
*   **Residuals Over Time:** Shows the regression residuals plotted against time. Ideally, these should be randomly scattered around zero.
*   **Residuals vs Fitted Values:** Plots residuals against the predicted values. A random scatter (no discernible pattern) indicates homoskedasticity. A funnel shape suggests heteroskedasticity.
*   **Q-Q Plot of Residuals:** Compares the distribution of residuals to a normal distribution. Points should ideally lie along the 45-degree line for normally distributed residuals.
*   **Residual Distribution (Histogram):** Shows the frequency distribution of residuals. Ideally, this should approximate a bell curve.

Alex reviews the diagnostic test results and the 4-panel plots for the selected stock. He notes that while Durbin-Watson and VIFs are generally acceptable for Fama-French factors, heteroskedasticity (Breusch-Pagan p-value < 0.05) is often present. This highlights the practical need for **HAC standard errors** in real-world reporting to ensure valid statistical inferences despite these common financial data characteristics.

## 5. Rolling Betas
Duration: 0:07:00

### Dynamic Factor Exposures: Rolling Betas
Static, full-sample betas can often mask how a stock's sensitivity to factors changes over time. This is particularly true during different market regimes, economic cycles, or significant geopolitical events. As a Portfolio Manager, Alex needs to understand this dynamic nature for effective risk management and tactical asset allocation. He will compute and visualize **rolling betas** over a defined window (e.g., 36 months) to observe how these exposures evolve.

This technique involves repeatedly running the factor regression (in this case, Fama-French 3-Factor) on a **moving window** of historical data. For example, a 36-month rolling window uses the most recent 36 months of data to calculate betas, then slides forward one month, recalculates, and so on. The resulting time series of betas provides invaluable insights into how the stock's "factor fingerprint" adapts to changing market conditions.

#### Your Turn: Calculate & Plot Rolling Betas
1.  Navigate to the "Rolling Betas" page.
2.  Ensure you have retrieved data.
3.  **Select a Stock:** Choose one of your stocks for rolling beta analysis from the dropdown.
4.  **Set Rolling Window Size:** Adjust the slider to select the number of months for the rolling window (e.g., 36 months).
5.  Click the **"Calculate & Plot Rolling Betas for [Selected Stock]"** button.

#### Interpreting Rolling Beta Time-Series Plots
The application will generate a plot showing the time series of the rolling Market, SMB, and HML betas for your selected stock.
*   Observe how the betas fluctuate over time. Are there periods where market beta increases, indicating higher sensitivity to the overall market?
*   Notice if the size (SMB) or value (HML) exposures change significantly. For example, a growth stock might show a consistently negative HML beta, but this could become less negative or even positive during a significant "value rally."
*   The plots include vertical shaded areas highlighting significant market events like the COVID-19 crash or periods of high inflation/rate hikes, allowing Alex to correlate changes in betas with economic shifts.

Alex analyzes the rolling beta plots for his selected stock. He observes how its market beta ($\beta_M$) might increase during periods of market stress, like the COVID-19 crash, indicating it becomes more sensitive to market movements during downturns. This dynamic view of factor exposures is critical for understanding the time-varying risk profile of his portfolio holdings.

## 6. Scenario Analysis
Duration: 0:08:00

### Forward-Looking Analysis: Scenario Projections
One of the most powerful applications of factor models for Alex is to project expected returns under various hypothetical macroeconomic scenarios. This shifts the analysis from purely backward-looking performance attribution to a forward-looking risk management and strategic planning tool.

By defining reasonable expected returns for the Fama-French factors in different economic environments, Alex can estimate how his target stocks might perform.

The scenario projection uses the estimated betas from the Fama-French 3-factor model:
$$ E[R_i - R_f] = \hat{\alpha}_i + \hat{\beta}_{i,M} E[R_m - R_f] + \hat{\beta}_{i,S} E[SMB] + \hat{\beta}_{i,H} E[HML] $$
where $E[...]$ denotes the expected value of the factors under a specific scenario (e.g., expected market excess return in a market crash), and $\hat{\alpha}$, $\hat{\beta}$ are the estimated coefficients (alpha and betas) from the full-sample regression.

Alex will define several plausible scenarios and then calculate the projected annualized excess return for each stock.

#### Your Turn: Define Scenarios and Project Returns
1.  Navigate to the "Scenario Analysis" page.
2.  Ensure you have run the Fama-French 3-Factor regressions.
3.  **Define Macroeconomic Scenarios:**
    *   You'll see a list of pre-defined scenarios (e.g., 'Base Case', 'Market Crash', 'Value Rotation').
    *   For each scenario, you can adjust the monthly expected returns for `Mkt-RF`, `SMB`, and `HML`. For example, in a 'Market Crash', you'd typically set `Mkt-RF` to a significantly negative value.
    *   You can also delete existing scenarios using the "Delete" button.
    *   **Add New Scenario:** Use the "Add New Scenario" expander to create your own scenario. Provide a name and set its expected factor returns.
4.  Click the **"Project Returns Under Scenarios"** button.

#### Reviewing Projected Returns
The application will generate a summary table showing the projected annualized excess returns for all your selected stocks under each defined scenario.

<aside class="console">
Example of Scenario Projections (partial):
Scenario         AAPL     BRK-B      TSLA       JNJ
Base Case       10.87%     7.65%     8.99%     6.42%
Market Crash   -15.53%   -10.21%   -21.78%   -12.05%
Value Rotation   3.12%    10.55%    -0.87%     8.88%
</aside>

Alex reviews the projected returns under different scenarios. He observes that in a 'Market Crash' scenario, `TSLA` (which typically has a high market beta) shows a significantly larger projected negative return compared to `JNJ` (which typically has a lower market beta). Conversely, in a 'Value Rotation' scenario, `BRK-B` (with a positive value beta) might be projected to perform relatively better than `AAPL` or `TSLA` (which often have negative value or growth tilts). This table provides Alex with critical insights for stress testing his portfolio, adjusting his risk exposure to specific factors, and informing his discussions with the investment committee.

## 7. Performance Attribution & Report
Duration: 0:10:00

### Performance Attribution and Model Summary
To complete his comprehensive analysis, Alex wants to visualize the contribution of each factor to the cumulative excess return of a stock. This "cumulative return decomposition" helps him attribute performance to market, size, and value factors versus the stock's idiosyncratic alpha. He also wants a visual comparison of the model's predicted versus actual returns and a final summary of all key metrics for easy reporting.

The cumulative contribution of each factor at time $T$ is given by:
$$ \text{Cumulative Factor Contribution}_X = \sum_{t=1}^T \hat{\beta}_{X} \cdot F_{X,t} $$
where $F_{X,t}$ is the factor return for factor $X$ (e.g., Mkt-RF, SMB, HML) at time $t$. The cumulative alpha contribution is $\sum_{t=1}^T \hat{\alpha}$, and the residual represents the unexplained portion.

#### Your Turn: Generate Attribution and Report
1.  Navigate to the "Performance Attribution & Report" page.
2.  Ensure you have run the Fama-French 3-Factor regressions and diagnostic tests for at least some stocks.
3.  **Select a Stock:** Choose one of your stocks for performance attribution.
4.  Click the **"Generate Performance Attribution for [Selected Stock]"** button.

#### Reviewing Performance Attribution Plots
The application will generate two key plots:

*   **Cumulative Return Decomposition Chart:** This stacked area chart breaks down the cumulative actual excess return of the selected stock into its components: Market Factor contribution, SMB Factor contribution, HML Factor contribution, Alpha contribution, and Residual (unexplained) contribution. The 'Actual Excess Return' is plotted as a black dashed line for comparison. This plot clearly shows how much of a stock's return is due to broad market movements, specific factor exposures, or genuinely idiosyncratic alpha (skill).
*   **Predicted vs. Actual Scatter Plot:** This scatter plot compares the model's predicted excess returns against the actual excess returns for each month. A 45-degree line represents perfect prediction. Points clustering closely around this line indicate a good model fit.

#### Comprehensive Factor Exposure & Performance Report
Finally, the application compiles a comprehensive table summarizing all the key findings for each stock, including:
*   Annualized FF3 Alpha and its p-value.
*   FF3 Betas (Market, SMB, HML).
*   FF3 R-squared and the improvement over CAPM R-squared.
*   Information Ratio.
*   Diagnostic test results (Durbin-Watson, Breusch-Pagan p-value, VIFs).

<aside class="console">
Example of Final Report (partial):
Stock  FF3_Alpha_Ann (%) FF3_Alpha_pvalue FF3_Beta_M FF3_Beta_S FF3_Beta_H FF3_R_squared R2_Improvement (FF3-CAPM) FF3_IR DW_Stat BP_Pvalue VIFs
AAPL               7.18%            0.005      1.050      0.297     -0.298         0.387                       0.033  0.536   1.916   0.0003  Mkt_RF: 1.05, SMB: 1.05, HML: 1.05
</aside>

Alex examines the cumulative return decomposition plots. This visualization is invaluable for presenting performance attribution to his investment committee, clearly showing how much of a stock's return is due to broad market movements, specific factor exposures, or genuinely idiosyncratic alpha. The predicted vs. actual plots provide a visual check on the model's fit. The final comprehensive report synthesizes all his findings, allowing him to quickly assess each stock's factor fingerprint, performance metrics, and the robustness of the model.

This workflow provides Alex with a powerful and reproducible Python solution, moving him away from manual spreadsheets and enabling deeper, more dynamic insights into his portfolio's risk and return drivers. This also serves as an interpretable baseline for him to compare more complex machine learning models in the future.
