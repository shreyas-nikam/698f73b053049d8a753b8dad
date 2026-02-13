
### 1. Application Overview

**Purpose of the Application**

This Streamlit application, "Factor Insights for Portfolio Managers," empowers CFA Charterholders and investment professionals like Alex to transition from traditional spreadsheet-based analysis to a robust, Python-driven workflow for factor modeling. It provides a comprehensive framework for estimating multi-factor regression models (CAPM, Fama-French 3-factor), performing performance attribution, conducting systematic risk management, and forecasting returns under hypothetical scenarios. The application emphasizes transparent, interpretable results, serving as a critical baseline for more advanced machine learning applications.

**High-Level Story Flow**

Alex, a Portfolio Manager, begins by defining the stocks and date range for his analysis.

1.  **Data Setup**: He initiates data retrieval and preparation, merging historical stock returns with Fama-French factors.
2.  **CAPM Baseline**: Alex establishes a single-factor baseline using the Capital Asset Pricing Model (CAPM) for each stock to understand basic market sensitivity and initial alpha.
3.  **Fama-French 3-Factor Model**: He then progresses to the Fama-French 3-Factor model to uncover deeper insights into market, size, and value factor exposures, comparing model explanatory power across stocks.
4.  **Model Diagnostics**: Critical diagnostic tests (autocorrelation, heteroskedasticity, multicollinearity) are performed to ensure the robustness and validity of the regression results, with clear interpretations and implications for financial time series data.
5.  **Rolling Betas**: To capture dynamic risk, Alex visualizes how factor exposures evolve over time using rolling regressions, noting shifts during significant market events.
6.  **Scenario Analysis**: He leverages the estimated factor betas for forward-looking risk management, projecting stock returns under various macroeconomic scenarios.
7.  **Performance Attribution & Report**: Finally, Alex generates a comprehensive "Factor Exposure & Risk Report," including cumulative return decomposition, predicted vs. actual returns, and a consolidated table of all key metrics, ready for presentation to his investment committee. This structured workflow provides him with a powerful tool for informed decision-making and performance explanation.

---

### 2. Code Requirements

```python
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm # Required for sm.qqplot and sm.add_constant in plotting/diagnostics

# Import all functions from the source.py file
from source import *

# Configure matplotlib for professional-looking plots (already done in source.py, but for local plotting context)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('deep')

# --- st.session_state Initialization ---
# Initialize session state variables if they don't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Introduction & Data Setup"
if 'tickers' not in st.session_state:
    st.session_state.tickers = ['AAPL', 'BRK-B', 'TSLA', 'JNJ']
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.date(2014, 1, 1)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.date(2024, 1, 1)
if 'df_merged' not in st.session_state:
    st.session_state.df_merged = None
if 'capm_results' not in st.session_state:
    st.session_state.capm_results = {}
if 'ff3_results' not in st.session_state:
    st.session_state.ff3_results = {}
if 'diagnostic_results' not in st.session_state:
    st.session_state.diagnostic_results = {}
if 'rolling_window_size' not in st.session_state:
    st.session_state.rolling_window_size = 36 # Default to 36 months
if 'macro_scenarios' not in st.session_state:
    st.session_state.macro_scenarios = {
        'Base Case': {'Mkt_RF': 0.08/12, 'SMB': 0.01/12, 'HML': 0.005/12},
        'Market Crash': {'Mkt_RF': -0.15/12, 'SMB': -0.05/12, 'HML': 0.02/12},
        'Value Rotation': {'Mkt_RF': 0.01/12, 'SMB': -0.01/12, 'HML': 0.08/12},
        'Small-Cap Rally': {'Mkt_RF': 0.03/12, 'SMB': 0.06/12, 'HML': 0.00/12},
        'Stagflation': {'Mkt_RF': -0.05/12, 'SMB': -0.02/12, 'HML': 0.05/12}
    }
if 'scenario_projections' not in st.session_state:
    st.session_state.scenario_projections = {}


# --- Sidebar Navigation ---
st.sidebar.title("Factor Insights Navigator")
page_options = [
    "Introduction & Data Setup",
    "CAPM Baseline",
    "Fama-French 3-Factor Model",
    "Model Diagnostics",
    "Rolling Betas",
    "Scenario Analysis",
    "Performance Attribution & Report"
]
st.session_state.current_page = st.sidebar.selectbox(
    "Go to page:",
    page_options,
    index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
)

# --- Page Rendering Logic ---

if st.session_state.current_page == "Introduction & Data Setup":
    st.title("📊 Factor Insights for Portfolio Managers")
    st.markdown(f"")
    st.markdown(f"**Persona:** Alex, a CFA Charterholder and Portfolio Manager at 'Alpha Investments', is transitioning to Python for more robust and scalable financial analysis.")
    st.markdown(f"")
    st.markdown(f"His goal is to move beyond simple market beta to multi-factor models for performance attribution, systematic risk management, and return forecasting.")
    st.markdown(f"")
    st.markdown(f"### 1. Data Acquisition and Preparation")
    st.markdown(f"Alex needs to gather historical monthly total returns for his target stocks and the widely-used Fama-French factor returns, along with the risk-free rate. This process, often manual and error-prone in spreadsheets, is automated and made reproducible in Python.")
    st.markdown(f"")
    st.markdown(f"He'll select a diversified set of stocks to analyze and merge them with the Fama-French data.")
    st.markdown(f"")
    st.markdown(f"**Practitioner Warning:** Fama-French factors use end-of-month dates while Yahoo Finance may use different conventions. Always verify that the merge is correct by spot-checking known dates (e.g., March 2020 COVID crash should show large negative Mkt-RF). A one-month misalignment would produce meaningless regressions. Also note that Fama-French returns are in percent (e.g., 2.5 = 2.5%) while Yahoo Finance returns are in decimal (0.025). Convert before merging.")
    st.markdown(f"")

    st.subheader("Configure Data Retrieval")
    tickers_input = st.text_input(
        "Enter stock tickers (comma-separated):",
        value=", ".join(st.session_state.tickers)
    )
    st.session_state.tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input(
            "Start Date:",
            value=st.session_state.start_date
        )
    with col2:
        st.session_state.end_date = st.date_input(
            "End Date:",
            value=st.session_state.end_date
        )

    if st.button("Retrieve and Prepare Data"):
        if st.session_state.tickers:
            with st.spinner("Fetching and merging data... This may take a moment."):
                try:
                    df_merged_temp = retrieve_and_merge_data(
                        st.session_state.tickers,
                        st.session_state.start_date.strftime('%Y-%m-%d'),
                        st.session_state.end_date.strftime('%Y-%m-%d')
                    )
                    st.session_state.df_merged = df_merged_temp
                    st.success("Data retrieved and prepared successfully!")
                    st.dataframe(st.session_state.df_merged.head())

                    # Plot a sample to verify alignment (e.g., first stock excess return vs Mkt-RF)
                    if st.session_state.tickers and not st.session_state.df_merged.empty:
                        first_ticker = st.session_state.tickers[0]
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(st.session_state.df_merged.index, st.session_state.df_merged[f'{first_ticker}_excess'], label=f'{first_ticker} Excess Return')
                        ax.plot(st.session_state.df_merged.index, st.session_state.df_merged['Mkt_RF'], label='Market Excess Return (Mkt-RF)')
                        ax.set_title(f'{first_ticker} Excess Return vs. Market Excess Return Over Time')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Monthly Return')
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig) # Close the figure to free memory

                except Exception as e:
                    st.error(f"Error retrieving or preparing data: {e}")
        else:
            st.warning("Please enter at least one stock ticker.")

    if st.session_state.df_merged is not None:
        st.markdown(f"Alex confirms the data is aligned and ready for model estimation. The dataset contains {st.session_state.df_merged.shape[0]} months of data.")


elif st.session_state.current_page == "CAPM Baseline":
    st.title("📈 CAPM Baseline: Understanding Market Sensitivity")
    st.markdown(f"### 2. Establishing a Baseline: The Capital Asset Pricing Model (CAPM)")
    st.markdown(f"Before diving into complex multi-factor models, Alex starts with the fundamental Capital Asset Pricing Model (CAPM) to establish a baseline understanding of each stock's sensitivity to the overall market. The CAPM is a single-factor model that explains the expected return of an asset based on its market risk.")

    st.markdown(r"$$ R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \epsilon_{i,t} $$")
    st.markdown(r"where $R_{i,t} - R_{f,t}$ is the excess return of asset $i$ at time $t$.")
    st.markdown(r"where $\alpha_i$ (Jensen's Alpha) is the asset's abnormal return not explained by the market factor. A positive and statistically significant $\alpha_i$ indicates outperformance after adjusting for market risk.")
    st.markdown(r"where $\beta_{i,M}$ (Market Beta) measures the asset's sensitivity to market movements. A $\beta_{i,M} > 1$ implies higher market sensitivity than the average stock, while $\beta_{i,M} < 1$ implies lower sensitivity.")
    st.markdown(r"where $R_{m,t} - R_{f,t}$ is the market excess return at time $t$.")
    st.markdown(r"where $\epsilon_{i,t}$ is the idiosyncratic error term.")
    st.markdown(f"")
    st.markdown(f"Alex will perform this regression for each of his target stocks to obtain their individual market betas and Jensen's alpha, along with statistical significance.")
    st.markdown(f"")

    if st.session_state.df_merged is not None and not st.session_state.df_merged.empty and st.session_state.tickers:
        if st.button("Run CAPM Regression for All Stocks"):
            st.session_state.capm_results = {}
            for stock in st.session_state.tickers:
                with st.spinner(f"Running CAPM for {stock}..."):
                    try:
                        results = run_capm_regression(st.session_state.df_merged, stock)
                        st.session_state.capm_results[stock] = results
                        st.subheader(f"CAPM Results for {stock}")
                        st.text(results['model'].summary())
                        st.markdown(f"**Interpretation for {stock}:**")
                        st.markdown(f"  - **Annualized Alpha:** {results['alpha_ann']:.2%} (p-value: {results['alpha_pval']:.4f})")
                        st.markdown(f"  - **Market Beta ($\beta_M$):** {results['beta_M']:.3f} (p-value: {results['beta_M_pval']:.4f})")
                        st.markdown(f"  - **R-squared:** {results['r_squared']:.3f}")
                        st.markdown(f"  - **Information Ratio:** {results['information_ratio']:.3f}")
                    except Exception as e:
                        st.error(f"Error running CAPM for {stock}: {e}")
            st.success("CAPM regressions completed!")
    else:
        st.warning("Please retrieve data on the 'Introduction & Data Setup' page first and ensure tickers are selected.")

    if st.session_state.capm_results:
        st.markdown(f"Alex reviews the outputs. He notes the market beta ($\beta_M$) values and pays close attention to Jensen's Alpha ($\alpha$) and its p-value. A high p-value for alpha (e.g., > 0.05) indicates that the stock's abnormal return is not statistically significant and could be due to random chance. The R-squared value tells him the proportion of the stock's excess return variance explained by the market factor.")


elif st.session_state.current_page == "Fama-French 3-Factor Model":
    st.title("🧠 Fama-French 3-Factor Model: Deeper Insights")
    st.markdown(f"### 3. Deeper Insights with Fama-French 3-Factor Model")
    st.markdown(f"While CAPM provides a basic understanding, Alex knows that investment performance is often driven by more than just market risk. He moves to the Fama-French 3-Factor Model, which adds size (SMB) and value (HML) factors, offering a richer explanation of asset returns and a more nuanced performance attribution.")

    st.markdown(r"$$ R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \beta_{i,S}SMB_t + \beta_{i,H}HML_t + \epsilon_{i,t} $$")
    st.markdown(r"where $\beta_{i,S}$ (Size Beta) measures the asset's exposure to the small-cap factor. A positive $\beta_{i,S}$ suggests a tilt towards smaller companies.")
    st.markdown(r"where $\beta_{i,H}$ (Value Beta) measures the asset's exposure to the value factor. A positive $\beta_{i,H}$ suggests a tilt towards value stocks (high book-to-market), while a negative $\beta_{i,H}$ indicates a growth stock tilt (low book-to-market).")
    st.markdown(r"where Other terms are as defined in the CAPM.")
    st.markdown(f"")
    st.markdown(f"Alex will run this model for all stocks, compare their factor exposures (their \"factor fingerprints\"), and quantify the incremental explanatory power of the additional factors (SMB and HML). He'll also use the Information Ratio, defined as:")
    st.markdown(r"$$ IR = \frac{\hat{\alpha}_{\text{ann}}}{\hat{\sigma}_{\epsilon}\sqrt{12}} = \frac{\text{Annualized Alpha}}{\text{Annualized Tracking Error}} $$")
    st.markdown(r"where The Information Ratio measures the risk-adjusted abnormal return, where $|IR| > 0.5$ is considered strong performance.")
    st.markdown(f"")

    if st.session_state.df_merged is not None and not st.session_state.df_merged.empty and st.session_state.tickers:
        if st.button("Run Fama-French 3-Factor Regression for All Stocks"):
            st.session_state.ff3_results = {}
            for stock in st.session_state.tickers:
                with st.spinner(f"Running FF3 for {stock}..."):
                    try:
                        results = run_ff3_regression(st.session_state.df_merged, stock)
                        st.session_state.ff3_results[stock] = results
                        st.subheader(f"Fama-French 3-Factor Results for {stock}")
                        st.text(results['model'].summary())
                    except Exception as e:
                        st.error(f"Error running FF3 for {stock}: {e}")
            st.success("Fama-French 3-Factor regressions completed!")
    else:
        st.warning("Please retrieve data on the 'Introduction & Data Setup' page first and ensure tickers are selected.")

    if st.session_state.capm_results and st.session_state.ff3_results and st.session_state.tickers:
        st.markdown(f"### Comparative Factor Exposure & Performance Table (CAPM vs. FF3)")
        summary_data = []
        for stock in st.session_state.tickers:
            capm_r = st.session_state.capm_results.get(stock)
            ff3_r = st.session_state.ff3_results.get(stock)

            if capm_r and ff3_r:
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
        st.dataframe(df_summary)

        st.markdown(f"Alex examines the outputs and the comparative table. He notices how stocks exhibit distinct factor fingerprints, like `TSLA`'s high $\\beta_M$ and negative $\\beta_H$ (growth tilt), versus `BRK-B`'s more moderate $\\beta_M$ and positive $\\beta_H$ (value tilt). The generally higher R-squared values for the FF3 model indicate improved explanatory power.")

        st.markdown(f"### Factor Beta Comparison Bar Chart (V2)")
        betas_df = pd.DataFrame({
            'Stock': st.session_state.tickers,
            'Beta_M': [st.session_state.ff3_results[s]['beta_M'] for s in st.session_state.tickers],
            'Beta_S': [st.session_state.ff3_results[s]['beta_S'] for s in st.session_state.tickers],
            'Beta_H': [st.session_state.ff3_results[s]['beta_H'] for s in st.session_state.tickers]
        })
        betas_melted = betas_df.melt(id_vars='Stock', var_name='Factor', value_name='Beta')
        fig_beta_comp, ax_beta_comp = plt.subplots(figsize=(14, 7))
        sns.barplot(x='Stock', y='Beta', hue='Factor', data=betas_melted, ax=ax_beta_comp)
        ax_beta_comp.set_title('Fama-French 3-Factor Betas Comparison Across Stocks')
        ax_beta_comp.set_xlabel('Stock')
        ax_beta_comp.set_ylabel('Factor Beta')
        ax_beta_comp.axhline(0, color='gray', linestyle='--')
        ax_beta_comp.legend(title='Factor', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig_beta_comp)
        plt.close(fig_beta_comp)

        st.markdown(f"### Security Market Line (SML) Plot (V1)")
        avg_excess_returns = st.session_state.df_merged[[f'{s}_excess' for s in st.session_state.tickers]].mean()
        sml_data = []
        for stock in st.session_state.tickers:
            sml_data.append({
                'Stock': stock,
                'Avg_Excess_Return': avg_excess_returns[f'{stock}_excess'],
                'Market_Beta': st.session_state.ff3_results[stock]['beta_M']
            })
        df_sml = pd.DataFrame(sml_data)
        avg_mkt_rf = st.session_state.df_merged['Mkt_RF'].mean()
        theoretical_sml_x = np.linspace(df_sml['Market_Beta'].min() * 0.8, df_sml['Market_Beta'].max() * 1.2, 100)
        theoretical_sml_y = theoretical_sml_x * avg_mkt_rf * 12

        fig_sml, ax_sml = plt.subplots(figsize=(12, 7))
        sns.scatterplot(x='Market_Beta', y='Avg_Excess_Return', hue='Stock', data=df_sml.mul(12), s=100, zorder=2, ax=ax_sml)
        ax_sml.plot(theoretical_sml_x, theoretical_sml_y, color='red', linestyle='--', label=f'Theoretical SML (E[Mkt-RF] Ann: {avg_mkt_rf*12:.2%})')
        ax_sml.set_title('Security Market Line (SML) Plot: Annualized Excess Returns vs. Market Beta')
        ax_sml.set_xlabel('Market Beta ($\\beta_M$)')
        ax_sml.set_ylabel('Annualized Average Excess Return')
        ax_sml.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax_sml.axvline(1, color='gray', linestyle=':', alpha=0.7, label='Market Beta = 1')
        ax_sml.legend()
        st.pyplot(fig_sml)
        plt.close(fig_sml)
    else:
        st.info("Run CAPM and Fama-French 3-Factor regressions to see comparative analysis and plots.")


elif st.session_state.current_page == "Model Diagnostics":
    st.title("🔬 Model Diagnostics: Ensuring Robustness")
    st.markdown(f"### 4. Validating Model Assumptions: Diagnostic Tests")
    st.markdown(f"Before relying on the factor model for critical investment decisions, Alex must perform diagnostic tests to check if the underlying assumptions of Ordinary Least Squares (OLS) regression are met. Violations of these assumptions (e.g., autocorrelation, heteroskedasticity, multicollinearity) can lead to inefficient or biased parameter estimates and incorrect statistical inferences (e.g., t-statistics, p-values). This step ensures the robustness of his analysis.")
    st.markdown(f"")

    st.markdown(f"He will check for:")
    st.markdown(f"")
    st.markdown(f"**Autocorrelation (Durbin-Watson statistic):** Checks if residuals are correlated over time.")
    st.markdown(r"$$ DW \approx 2(1 - \rho_1) $$")
    st.markdown(r"where $\rho_1$ is the first-order autocorrelation of residuals. A value close to 2 indicates no autocorrelation. For financial time series, positive autocorrelation ($DW < 2$, especially $< 1.5$) can indicate momentum effects or missing factors.")
    st.markdown(f"")
    st.markdown(f"**Heteroskedasticity (Breusch-Pagan test):** Checks if the variance of the residuals is constant across all levels of independent variables. Heteroskedasticity (Breusch-Pagan p-value $< 0.05$) leads to inefficient estimates and incorrect standard errors.")
    st.markdown(f"")
    st.markdown(f"**Multicollinearity (Variance Inflation Factor - VIF):** Checks if independent variables are highly correlated with each other.")
    st.markdown(r"$$ VIF_j = \frac{1}{1 - R_j^2} $$")
    st.markdown(r"where $R_j^2$ is the R-squared from regressing factor $j$ on all other factors. High multicollinearity ($VIF > 5$ or $10$) can make coefficient estimates unstable and difficult to interpret.")
    st.markdown(f"")
    st.markdown(f"**Practitioner Warning:** Heteroskedasticity is common in financial data. If the Breusch-Pagan test rejects homoskedasticity (typical for equity returns, where volatility clusters in crises), switch to Newey-West HAC standard errors for robust inference. HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors produce valid t-statistics and p-values even when classical OLS assumptions are violated—a critical technique for financial time-series.")
    st.markdown(f"")

    if st.session_state.ff3_results and st.session_state.tickers:
        selected_stock = st.selectbox(
            "Select stock for diagnostic tests:",
            st.session_state.tickers,
            key='diag_stock_select'
        )

        if st.button(f"Run Diagnostics for {selected_stock}"):
            ff3_model = st.session_state.ff3_results[selected_stock]['model']
            # X_ff3 needs to exactly match the exogenous variables used in the FF3 model
            # This typically involves `sm.add_constant(df_merged[['Mkt_RF', 'SMB', 'HML']])`
            # The run_diagnostic_tests function handles extracting model.exog, which is correct
            # Re-creating X for VIF calculation for clarity
            X_ff3_for_vif = st.session_state.df_merged[['Mkt_RF', 'SMB', 'HML']] # Exogenous variables without constant

            with st.spinner(f"Running diagnostic tests for {selected_stock}'s FF3 model..."):
                try:
                    results = run_diagnostic_tests(ff3_model, X_ff3_for_vif, selected_stock)
                    st.session_state.diagnostic_results[selected_stock] = results

                    st.subheader(f"Diagnostic Test Results for {selected_stock}")
                    st.markdown(f"  - **Durbin-Watson statistic:** {results['dw_stat']:.3f} - {results['dw_interpretation']}")
                    st.markdown(f"  - **Breusch-Pagan p-value:** {results['bp_pvalue']:.4f} - {results['bp_interpretation']}")
                    vif_str = ", ".join([f"{k}: {v:.2f}" for k, v in results['vif_results'].items()])
                    st.markdown(f"  - **VIF results:** {vif_str}")
                    st.markdown(f"  - **VIF interpretation:** {results['vif_interpretation']}")

                    st.markdown(f"### Diagnostic 4-Panel Plot of Residuals (V3)")
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    fig.suptitle(f'Regression Diagnostic Plots for {selected_stock} (Fama-French 3-Factor Model)', fontsize=16)

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

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Error running diagnostic tests for {selected_stock}: {e}")
            st.success("Diagnostic tests completed!")
    else:
        st.warning("Please run Fama-French 3-Factor regressions on the previous page first.")

    # Display interpretation even if the button wasn't clicked, if results exist
    if st.session_state.diagnostic_results.get(selected_stock):
        st.markdown(f"Alex reviews the diagnostic test results and the 4-panel plots for {selected_stock}. He notes that while Durbin-Watson and VIFs are generally acceptable for Fama-French factors, heteroskedasticity (Breusch-Pagan p-value < 0.05) is often present. This highlights the need for **HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors** in real-world reporting to ensure valid statistical inferences despite these common financial data characteristics.")


elif st.session_state.current_page == "Rolling Betas":
    st.title("🔄 Rolling Betas: Capturing Dynamic Risk")
    st.markdown(f"### 5. Dynamic Factor Exposures: Rolling Betas")
    st.markdown(f"Static, full-sample betas can mask how a stock's sensitivity to factors changes over time, especially during different market regimes or significant economic events. As a Portfolio Manager, Alex needs to understand this dynamic nature for effective risk management and tactical asset allocation. He will compute and visualize rolling betas over a defined window (e.g., 36 months) to observe how these exposures evolve.")
    st.markdown(f"")
    st.markdown(f"This technique involves running the factor regression repeatedly on a moving window of historical data. The resulting time series of betas provides insights into how the stock's \"factor fingerprint\" adapts to changing market conditions.")
    st.markdown(f"")

    if st.session_state.df_merged is not None and not st.session_state.df_merged.empty and st.session_state.tickers:
        selected_stock_rb = st.selectbox(
            "Select stock for rolling beta analysis:",
            st.session_state.tickers,
            key='rolling_beta_stock_select'
        )
        st.session_state.rolling_window_size = st.slider(
            "Select Rolling Window Size (months):",
            min_value=12,
            max_value=60,
            value=st.session_state.rolling_window_size,
            step=6
        )

        if st.button(f"Calculate & Plot Rolling Betas for {selected_stock_rb}"):
            with st.spinner(f"Calculating rolling {st.session_state.rolling_window_size}-month betas for {selected_stock_rb}..."):
                try:
                    rolling_betas_df = calculate_rolling_betas(
                        st.session_state.df_merged,
                        selected_stock_rb,
                        st.session_state.rolling_window_size
                    )
                    
                    st.markdown(f"### Rolling Beta Time-Series Plot (V4)")
                    fig, ax = plt.subplots(figsize=(14, 7))
                    rolling_betas_df.plot(ax=ax)
                    
                    ax.set_title(f'Rolling {st.session_state.rolling_window_size}-Month Fama-French Factor Betas for {selected_stock_rb}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Beta Value')
                    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
                    
                    # Annotate significant market events (approximate dates for illustration)
                    plot_start = rolling_betas_df.index.min()
                    plot_end = rolling_betas_df.index.max()

                    if pd.Timestamp('2020-02-01') >= plot_start and pd.Timestamp('2020-04-01') <= plot_end:
                        ax.axvspan(datetime.datetime(2020, 2, 1), datetime.datetime(2020, 4, 1), color='red', alpha=0.2, label='COVID-19 Crash')
                    if pd.Timestamp('2022-01-01') >= plot_start and pd.Timestamp('2022-12-01') <= plot_end:
                        ax.axvspan(datetime.datetime(2022, 1, 1), datetime.datetime(2022, 12, 1), color='purple', alpha=0.1, label='Inflation/Rate Hikes')
                    
                    ax.legend(title='Factor', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Error calculating or plotting rolling betas for {selected_stock_rb}: {e}")
            st.success("Rolling beta analysis completed!")
    else:
        st.warning("Please retrieve data on the 'Introduction & Data Setup' page first and ensure tickers are selected.")

    if selected_stock_rb and st.session_state.df_merged is not None:
        st.markdown(f"Alex analyzes the rolling beta plots for {selected_stock_rb}. He observes how its market beta ($\beta_M$) might increase during periods of market stress, like the COVID-19 crash, indicating it becomes more sensitive to market movements during downturns. This dynamic view of factor exposures is critical for understanding the time-varying risk profile of his portfolio holdings.")


elif st.session_state.current_page == "Scenario Analysis":
    st.title("🔮 Scenario Analysis: Forward-Looking Projections")
    st.markdown(f"### 6. Forward-Looking Analysis: Scenario Projections")
    st.markdown(f"One of the most powerful applications of factor models for Alex is to project expected returns under various hypothetical macroeconomic scenarios. This shifts the analysis from purely backward-looking performance attribution to a forward-looking risk management and strategic planning tool.")
    st.markdown(f"")
    st.markdown(f"By defining reasonable expected returns for the Fama-French factors in different economic environments, Alex can estimate how his target stocks might perform.")
    st.markdown(f"The scenario projection uses the estimated betas from the Fama-French 3-factor model:")
    st.markdown(r"$$ E[R_i - R_f] = \hat{\alpha}_i + \hat{\beta}_{i,M} E[R_m - R_f] + \hat{\beta}_{i,S} E[SMB] + \hat{\beta}_{i,H} E[HML] $$")
    st.markdown(r"where $E[...]$ denotes the expected value of the factors under a specific scenario, and $\hat{\alpha}$, $\hat{\beta}$ are the estimated coefficients from the full-sample regression.")
    st.markdown(f"Alex will define several plausible scenarios and then calculate the projected annualized excess return for each stock.")
    st.markdown(f"")

    if st.session_state.ff3_results and st.session_state.tickers:
        st.subheader("Define Macroeconomic Scenarios (Monthly Expected Factor Returns)")

        # Display current scenarios and allow modification/addition
        current_scenarios_copy = st.session_state.macro_scenarios.copy() # Operate on a copy to allow deletion/addition
        scenarios_to_delete = []

        for scenario_name in current_scenarios_copy.keys():
            st.markdown(f"**Scenario: {scenario_name}**")
            col_mkt, col_smb, col_hml, col_del = st.columns([0.3, 0.3, 0.3, 0.1])
            with col_mkt:
                current_scenarios_copy[scenario_name]['Mkt_RF'] = st.number_input(f"Mkt-RF (monthly):", value=current_scenarios_copy[scenario_name]['Mkt_RF'], format="%.4f", key=f"{scenario_name}_Mkt_RF")
            with col_smb:
                current_scenarios_copy[scenario_name]['SMB'] = st.number_input(f"SMB (monthly):", value=current_scenarios_copy[scenario_name]['SMB'], format="%.4f", key=f"{scenario_name}_SMB")
            with col_hml:
                current_scenarios_copy[scenario_name]['HML'] = st.number_input(f"HML (monthly):", value=current_scenarios_copy[scenario_name]['HML'], format="%.4f", key=f"{scenario_name}_HML")
            with col_del:
                if st.button("Delete", key=f"delete_{scenario_name}"):
                    scenarios_to_delete.append(scenario_name)
            st.markdown("---")
        
        # Apply deletions
        if scenarios_to_delete:
            for sc_name in scenarios_to_delete:
                del st.session_state.macro_scenarios[sc_name]
            st.success("Scenario(s) deleted. Please re-run projection.")
            st.experimental_rerun() # Rerun to update the display after deletion

        # Update original session state after modifications
        st.session_state.macro_scenarios = current_scenarios_copy
        
        # Option to add a new scenario - simplified
        with st.expander("Add New Scenario"):
            new_scenario_name = st.text_input("New Scenario Name:", key="new_scenario_name")
            new_mkt = st.number_input("New Mkt-RF (monthly):", value=0.005, format="%.4f", key="new_mkt")
            new_smb = st.number_input("New SMB (monthly):", value=0.001, format="%.4f", key="new_smb")
            new_hml = st.number_input("New HML (monthly):", value=0.000, format="%.4f", key="new_hml")
            if st.button("Add Scenario") and new_scenario_name:
                if new_scenario_name not in st.session_state.macro_scenarios:
                    st.session_state.macro_scenarios[new_scenario_name] = {'Mkt_RF': new_mkt, 'SMB': new_smb, 'HML': new_hml}
                    st.success(f"Scenario '{new_scenario_name}' added!")
                    st.experimental_rerun() # Rerun to display new scenario in list
                else:
                    st.warning(f"Scenario '{new_scenario_name}' already exists. Please choose a different name.")

        if st.button("Project Returns Under Scenarios"):
            st.session_state.scenario_projections = {}
            for stock in st.session_state.tickers:
                ff3_model_params = st.session_state.ff3_results[stock]['model'].params
                projections = project_returns_under_scenarios(ff3_model_params, st.session_state.macro_scenarios)
                st.session_state.scenario_projections[stock] = projections
            
            st.success("Scenario projections completed!")
            
            st.markdown(f"### Summary of Projected Annualized Excess Returns Across All Stocks")
            all_projections_df = pd.DataFrame()
            for stock, projections in st.session_state.scenario_projections.items():
                projections_indexed = projections.set_index('Scenario').rename(columns={'Projected_Annual_Excess_Return': stock})
                if all_projections_df.empty:
                    all_projections_df = projections_indexed
                else:
                    all_projections_df = all_projections_df.join(projections_indexed)
            
            st.dataframe(all_projections_df.applymap(lambda x: f"{x:.2%}"))

    else:
        st.warning("Please run Fama-French 3-Factor regressions on the 'Fama-French 3-Factor Model' page first and ensure tickers are selected.")

    if st.session_state.scenario_projections:
        st.markdown(f"Alex reviews the projected returns under different scenarios. He observes that in a 'Market Crash' scenario, `TSLA` (high market beta) shows a significantly larger projected negative return compared to `JNJ` (lower market beta). In a 'Value Rotation' scenario, `BRK-B` (positive value beta) might be projected to perform relatively better than `AAPL` or `TSLA` (negative value/growth tilt). This table provides Alex with critical insights for stress testing his portfolio, adjusting his risk exposure to specific factors, and informing his discussions with the investment committee.")


elif st.session_state.current_page == "Performance Attribution & Report":
    st.title("🏆 Performance Attribution & Report")
    st.markdown(f"### 7. Performance Attribution and Model Summary")
    st.markdown(f"To complete his comprehensive analysis, Alex wants to visualize the contribution of each factor to the cumulative excess return of a stock. This \"cumulative return decomposition\" helps him attribute performance to market, size, and value factors versus the stock's idiosyncratic alpha. He also wants a visual comparison of the model's predicted versus actual returns and a final summary of all key metrics for easy reporting.")
    st.markdown(f"")
    st.markdown(f"The cumulative contribution of each factor at time $T$ is given by:")
    st.markdown(r"$$ \text{Cumulative Factor Contribution}_X = \sum_{t=1}^T \hat{\beta}_{X} \cdot F_{X,t} $$")
    st.markdown(r"where $F_{X,t}$ is the factor return for factor $X$ at time $t$. The cumulative alpha contribution is $\sum_{t=1}^T \hat{\alpha}$.")
    st.markdown(f"")

    if st.session_state.ff3_results and st.session_state.df_merged is not None and st.session_state.tickers:
        selected_stock_pa = st.selectbox(
            "Select stock for performance attribution:",
            st.session_state.tickers,
            key='perf_attr_stock_select'
        )

        if st.button(f"Generate Performance Attribution for {selected_stock_pa}"):
            ff3_model = st.session_state.ff3_results[selected_stock_pa]['model']
            ff3_model_params = ff3_model.params

            st.markdown(f"### Cumulative Return Decomposition Chart (V6)")
            # Inlining the plotting logic from source.py's plot_cumulative_return_decomposition
            y = st.session_state.df_merged[f'{selected_stock_pa}_excess']
            X = st.session_state.df_merged[['Mkt_RF', 'SMB', 'HML']]

            alpha = ff3_model_params['const']
            beta_M = ff3_model_params['Mkt_RF']
            beta_S = ff3_model_params['SMB']
            beta_H = ff3_model_params['HML']

            market_contribution = beta_M * X['Mkt_RF']
            smb_contribution = beta_S * X['SMB']
            hml_contribution = beta_H * X['HML']
            alpha_contribution = pd.Series(alpha, index=st.session_state.df_merged.index)
            total_model_return = market_contribution + smb_contribution + hml_contribution + alpha_contribution
            epsilon_contribution = y - total_model_return

            attribution_df = pd.DataFrame({
                'Market Factor': market_contribution.cumsum(),
                'SMB Factor': smb_contribution.cumsum(),
                'HML Factor': hml_contribution.cumsum(),
                'Alpha': alpha_contribution.cumsum(),
                'Residual (Unexplained)': epsilon_contribution.cumsum(),
                'Actual Excess Return': y.cumsum()
            }, index=st.session_state.df_merged.index)

            fig_cum_decomp, ax_cum_decomp = plt.subplots(figsize=(14, 7))
            attribution_df[['Market Factor', 'SMB Factor', 'HML Factor', 'Alpha', 'Residual (Unexplained)']].plot(
                kind='area', stacked=True, ax=ax_cum_decomp, alpha=0.7
            )
            attribution_df['Actual Excess Return'].plot(ax=ax_cum_decomp, color='black', linestyle='--', linewidth=2, label='Actual Excess Return')

            ax_cum_decomp.set_title(f'Cumulative Return Decomposition for {selected_stock_pa} (Fama-French 3-Factor Model)')
            ax_cum_decomp.set_xlabel('Date')
            ax_cum_decomp.set_ylabel('Cumulative Return')
            ax_cum_decomp.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig_cum_decomp)
            plt.close(fig_cum_decomp)

            st.markdown(f"### Predicted vs. Actual Scatter Plot (V5)")
            # Inlining the plotting logic from source.py's plot_predicted_vs_actual
            y_actual = st.session_state.df_merged[f'{selected_stock_pa}_excess']
            X_pred = sm.add_constant(st.session_state.df_merged[['Mkt_RF', 'SMB', 'HML']])
            y_predicted = ff3_model.predict(X_pred)

            fig_pred_actual, ax_pred_actual = plt.subplots(figsize=(8, 8))
            sns.scatterplot(x=y_predicted, y=y_actual, alpha=0.6, ax=ax_pred_actual)
            min_val = min(y_actual.min(), y_predicted.min())
            max_val = max(y_actual.max(), y_predicted.max())
            ax_pred_actual.plot([min_val, max_val], [min_val, max_val],
                                color='red', linestyle='--', label='45-degree line (Perfect Prediction)')
            ax_pred_actual.set_title(f'Predicted vs. Actual Excess Returns for {selected_stock_pa}')
            ax_pred_actual.set_xlabel('Predicted Excess Return')
            ax_pred_actual.set_ylabel('Actual Excess Return')
            ax_pred_actual.legend()
            ax_pred_actual.grid(True)
            plt.tight_layout()
            st.pyplot(fig_pred_actual)
            plt.close(fig_pred_actual)
            st.success("Performance attribution plots generated!")

    else:
        st.warning("Please run Fama-French 3-Factor regressions on the 'Fama-French 3-Factor Model' page first and ensure tickers are selected.")

    if st.session_state.ff3_results and st.session_state.capm_results and st.session_state.diagnostic_results and st.session_state.tickers:
        st.markdown(f"### Comprehensive Factor Exposure & Performance Report")
        final_summary_data = []
        for stock in st.session_state.tickers:
            capm_r = st.session_state.capm_results.get(stock)
            ff3_r = st.session_state.ff3_results.get(stock)
            diag_r = st.session_state.diagnostic_results.get(stock)

            if capm_r and ff3_r and diag_r:
                r_squared_improvement = ff3_r['r_squared'] - capm_r['r_squared']
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
        
        # Format for better readability
        df_final_report_formatted = df_final_report.copy()
        for col in ['FF3_Alpha_Ann (%)']:
            if col in df_final_report_formatted.columns:
                df_final_report_formatted[col] = df_final_report_formatted[col].map('{:.2f}%'.format)
        for col in ['FF3_Alpha_pvalue', 'FF3_Beta_M', 'FF3_Beta_S', 'FF3_Beta_H', 'FF3_R_squared', 'R2_Improvement (FF3-CAPM)', 'FF3_IR', 'DW_Stat']:
            if col in df_final_report_formatted.columns:
                df_final_report_formatted[col] = df_final_report_formatted[col].map('{:.3f}'.format)
        if 'BP_Pvalue' in df_final_report_formatted.columns:
            df_final_report_formatted['BP_Pvalue'] = df_final_report_formatted['BP_Pvalue'].map('{:.4f}'.format)

        st.dataframe(df_final_report_formatted)

        st.markdown(f"Alex examines the cumulative return decomposition plots. This visualization is invaluable for presenting performance attribution to his investment committee, clearly showing how much of a stock's return is due to broad market movements, specific factor exposures, or genuinely idiosyncratic alpha (skill). The predicted vs. actual plots provide a visual check on the model's fit. The final comprehensive report synthesizes all his findings, allowing him to quickly assess each stock's factor fingerprint, performance metrics, and the robustness of the model.")
        st.markdown(f"")
        st.markdown(f"This workflow provides Alex with a powerful and reproducible Python workflow, moving him away from manual spreadsheets and enabling deeper, more dynamic insights into his portfolio's risk and return drivers. This also serves as an interpretable baseline for him to compare more complex machine learning models in the future.")
```
