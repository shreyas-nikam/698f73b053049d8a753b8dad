import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import io
import zipfile
import requests

# Configure matplotlib for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('deep')

# --- Core Data Retrieval and Regression Functions ---


def load_ff3_monthly_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load Ken French F-F Research Data Factors (Monthly) directly from Dartmouth.
    Returns decimals (not percent) with columns: Mkt_RF, SMB, HML, RF.
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # Usually contains a single CSV file
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw = zf.read(csv_name).decode("latin-1").splitlines()

    # Find start of monthly table (line after the header rows)
    # The file typically has a few header lines, then:
    # " 192607   2.96  -2.30  -2.87   0.22"
    # and later " Annual Factors: January-December "
    start_idx = None
    end_idx = None

    for i, line in enumerate(raw):
        if line.strip().startswith("1926") or line.strip().startswith("1927"):
            start_idx = i
            break

    for i, line in enumerate(raw):
        if "Annual Factors" in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise ValueError("Could not parse Ken French factor file format.")

    data_lines = raw[start_idx:end_idx]

    rows = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue

        # Split on comma if present, else whitespace
        parts = [p.strip()
                 for p in (line.split(",") if "," in line else line.split())]

        # Need at least YYYYMM + 4 factor columns
        if len(parts) < 5:
            continue

        yyyymm = parts[0]

        # Keep only true monthly rows like "192607"
        if not (yyyymm.isdigit() and len(yyyymm) == 6):
            continue

        rows.append(parts[:5])

    df = pd.DataFrame(rows, columns=["YYYYMM", "Mkt_RF", "SMB", "HML", "RF"])

    # Convert to numeric (coerce bad values to NaN) then drop any incomplete rows
    for c in ["Mkt_RF", "SMB", "HML", "RF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    # Convert YYYYMM to month-end timestamp
    df["Date"] = pd.to_datetime(df["YYYYMM"], format="%Y%m")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")
    df = df.drop(columns=["YYYYMM"]).set_index("Date")

    # Percent -> decimal
    df = df / 100.0

    # Filter date range
    df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    return df


def retrieve_and_merge_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
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
    print("--- Retrieving Fama-French Factors (Ken French) ---")
    ff_data = load_ff3_monthly_factors(start_date, end_date)

    print("--- Retrieving Stock Returns ---")
    stock_prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # If single ticker, yfinance returns a Series -> convert to DataFrame
    if isinstance(stock_prices, pd.Series):
        stock_prices = stock_prices.to_frame(name=tickers[0])

    # Convert daily prices -> month-end prices to match Ken French month-end dating
    stock_prices = stock_prices.resample("M").last()

    stock_returns = stock_prices.pct_change().dropna()
    stock_returns.index.name = "Date"

    print("--- Merging Data ---")
    merged_data = stock_returns.join(ff_data, how='inner')

    for ticker in tickers:
        merged_data[f'{ticker}_excess'] = merged_data[ticker] - \
            merged_data['RF']

    columns_for_analysis = [
        f'{t}_excess' for t in tickers] + ['Mkt_RF', 'SMB', 'HML'] + ['RF']
    merged_data = merged_data[columns_for_analysis].dropna()

    print(
        f"Merged dataset shape: {merged_data.shape[0]} months, {merged_data.shape[1]} columns.")
    return merged_data


def run_capm_regression(df: pd.DataFrame, stock_ticker: str) -> dict:
    """
    Runs CAPM regression for a given stock and extracts key metrics.

    Args:
        df (pd.DataFrame): Merged DataFrame containing stock excess returns and Mkt_RF.
        stock_ticker (str): The ticker symbol of the stock to analyze.

    Returns:
        dict: A dictionary containing CAPM regression results including the model object.
    """
    y = df[f'{stock_ticker}_excess']
    X = sm.add_constant(df['Mkt_RF'])  # Add constant for alpha

    model = sm.OLS(y, X).fit()

    alpha = model.params['const']
    beta_M = model.params['Mkt_RF']
    alpha_pval = model.pvalues['const']
    beta_M_pval = model.pvalues['Mkt_RF']
    r_squared = model.rsquared
    resid_std = model.resid.std()

    alpha_ann = alpha * 12
    resid_std_ann = resid_std * np.sqrt(12)
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


def run_ff3_regression(df: pd.DataFrame, stock_ticker: str) -> dict:
    """
    Runs Fama-French 3-factor regression for a given stock and extracts key metrics.

    Args:
        df (pd.DataFrame): Merged DataFrame containing stock excess returns and FF factors.
        stock_ticker (str): The ticker symbol of the stock to analyze.

    Returns:
        dict: A dictionary containing FF3 regression results including the model object.
    """
    y = df[f'{stock_ticker}_excess']
    X = sm.add_constant(df[['Mkt_RF', 'SMB', 'HML']])  # Add constant for alpha

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

    alpha_ann = alpha * 12
    resid_std_ann = resid_std * np.sqrt(12)
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


def run_diagnostic_tests(model: sm.regression.linear_model.RegressionResultsWrapper, X_df: pd.DataFrame) -> dict:
    """
    Performs diagnostic tests for an OLS regression model.

    Args:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted OLS model.
        X_df (pd.DataFrame): DataFrame of independent variables (exogenous).

    Returns:
        dict: A dictionary containing diagnostic test results.
    """
    dw_stat = durbin_watson(model.resid)
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_pvalue = bp_test[1]

    X_for_vif = X_df.drop(columns='const', errors='ignore')
    vif_results = {}
    if not X_for_vif.empty and X_for_vif.shape[1] > 0:
        for i, col in enumerate(X_for_vif.columns):
            vif = variance_inflation_factor(X_for_vif.values, i)
            vif_results[col] = vif
    else:
        # If only constant or single factor, VIF is not typically interpreted for multicollinearity
        # or it's 1 for a single factor.
        # Likely CAPM with const + Mkt_RF
        if 'Mkt_RF' in X_df.columns and X_df.shape[1] == 2:
            # VIF is 1 for a single independent variable
            vif_results['Mkt_RF'] = 1.0
        else:
            vif_results['No_Applicable_Factors_for_VIF'] = np.nan

    dw_interpretation = "No significant autocorrelation (around 2)." if 1.5 <= dw_stat <= 2.5 else "Potential autocorrelation issues (outside 1.5-2.5 range)."
    bp_interpretation = "Homoskedasticity assumption likely holds (p-value >= 0.05)." if bp_pvalue >= 0.05 else "Heteroskedasticity detected (p-value < 0.05). Consider HAC standard errors."
    vif_interpretation = "No problematic multicollinearity (VIFs < 5)."
    if any(v > 5 for v in vif_results.values() if not np.isnan(v)):
        vif_interpretation = "Problematic multicollinearity detected (VIF > 5 for some factors)."

    return {
        'dw_stat': dw_stat,
        'dw_interpretation': dw_interpretation,
        'bp_pvalue': bp_pvalue,
        'bp_interpretation': bp_interpretation,
        'vif_results': vif_results,
        'vif_interpretation': vif_interpretation
    }


def calculate_rolling_betas(df: pd.DataFrame, stock_ticker: str, window_size: int = 36) -> pd.DataFrame:
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
        window_df = df.iloc[i - window_size + 1: i + 1]

        y_roll = window_df[y_col]
        X_roll = sm.add_constant(window_df[X_cols])

        if len(y_roll) > len(X_cols) + 1 and not X_roll.isnull().any().any():
            try:
                model_roll = sm.OLS(y_roll, X_roll).fit()
                if all(col in model_roll.params for col in X_cols):
                    rolling_betas.loc[df.index[i], [
                        'Beta_M', 'Beta_S', 'Beta_H']] = model_roll.params[X_cols].values
                else:
                    rolling_betas.loc[df.index[i], [
                        'Beta_M', 'Beta_S', 'Beta_H']] = np.nan
            except (ValueError, np.linalg.LinAlgError):
                rolling_betas.loc[df.index[i], [
                    'Beta_M', 'Beta_S', 'Beta_H']] = np.nan
        else:
            rolling_betas.loc[df.index[i], [
                'Beta_M', 'Beta_S', 'Beta_H']] = np.nan

    return rolling_betas.astype(float)


def project_returns_under_scenarios(ff3_model_params: pd.Series, scenarios: dict) -> pd.DataFrame:
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

    alpha = ff3_model_params['const']
    beta_M = ff3_model_params['Mkt_RF']
    beta_S = ff3_model_params['SMB']
    beta_H = ff3_model_params['HML']

    for scenario_name, factor_returns in scenarios.items():
        exp_ret_monthly = (alpha +
                           beta_M * factor_returns.get('Mkt_RF', 0) +
                           beta_S * factor_returns.get('SMB', 0) +
                           beta_H * factor_returns.get('HML', 0))

        exp_ret_annual = exp_ret_monthly * 12
        projected_returns_data.append(
            {'Scenario': scenario_name, 'Projected_Annual_Excess_Return': exp_ret_annual})

    return pd.DataFrame(projected_returns_data)

# --- Plotting Helper Functions ---


def plot_data_alignment(df: pd.DataFrame, stock_ticker: str):
    """
    Plots a sample of stock excess return vs. market excess return to verify data alignment.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[f'{stock_ticker}_excess'],
             label=f'{stock_ticker} Excess Return')
    plt.plot(df.index, df['Mkt_RF'], label='Market Excess Return (Mkt-RF)')
    plt.title(f'{stock_ticker} Excess Return vs. Market Excess Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Monthly Return')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_factor_betas_comparison(tickers: list, ff3_results: dict):
    """
    Creates a bar chart comparing Fama-French 3-factor betas across multiple stocks.
    """
    betas_df = pd.DataFrame({
        'Stock': tickers,
        'Beta_M': [ff3_results[s]['beta_M'] for s in tickers],
        'Beta_S': [ff3_results[s]['beta_S'] for s in tickers],
        'Beta_H': [ff3_results[s]['beta_H'] for s in tickers]
    })

    betas_melted = betas_df.melt(
        id_vars='Stock', var_name='Factor', value_name='Beta')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Stock', y='Beta', hue='Factor', data=betas_melted)
    plt.title('Fama-French 3-Factor Betas Comparison Across Stocks')
    plt.xlabel('Stock')
    plt.ylabel('Factor Beta')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend(title='Factor', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_sml_analysis(df_merged: pd.DataFrame, tickers: list, ff3_results: dict):
    """
    Plots the Security Market Line (SML) for a set of stocks based on FF3 Market Beta.
    """
    avg_excess_returns = df_merged[[f'{s}_excess' for s in tickers]].mean()

    sml_data = []
    for stock in tickers:
        sml_data.append({
            'Stock': stock,
            'Avg_Excess_Return': avg_excess_returns[f'{stock}_excess'],
            'Market_Beta': ff3_results[stock]['beta_M']
        })
    df_sml = pd.DataFrame(sml_data)

    avg_mkt_rf = df_merged['Mkt_RF'].mean()
    theoretical_sml_x = np.linspace(
        df_sml['Market_Beta'].min() * 0.8, df_sml['Market_Beta'].max() * 1.2, 100)
    theoretical_sml_y = theoretical_sml_x * avg_mkt_rf * 12

    plt.figure(figsize=(12, 7))
    df_sml_plot = df_sml.copy()
    df_sml_plot["Avg_Excess_Return_Ann"] = df_sml_plot["Avg_Excess_Return"] * 12

    sns.scatterplot(
        x="Market_Beta",
        y="Avg_Excess_Return_Ann",
        hue="Stock",
        data=df_sml_plot,
        s=100,
        zorder=2,
    )

    plt.plot(theoretical_sml_x, theoretical_sml_y, color='red', linestyle='--',
             label=f'Theoretical SML (E[Mkt-RF] Ann: {avg_mkt_rf*12:.2%})')
    plt.title(
        'Security Market Line (SML) Plot: Annualized Excess Returns vs. Market Beta')
    plt.xlabel(r'Market Beta ($\beta_M$)')
    plt.ylabel('Annualized Average Excess Return')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(1, color='gray', linestyle=':',
                alpha=0.7, label='Market Beta = 1')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_regression_diagnostics(ff3_model: sm.regression.linear_model.RegressionResultsWrapper, stock_ticker: str):
    """
    Generates a 4-panel diagnostic plot for an OLS regression model's residuals.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f'Regression Diagnostic Plots for {stock_ticker} (Fama-French 3-Factor Model)', fontsize=16)

    axes[0, 0].plot(ff3_model.resid.index, ff3_model.resid)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_ylabel('Residual Value')

    axes[0, 1].scatter(ff3_model.fittedvalues, ff3_model.resid, alpha=0.5)
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_title('Residuals vs Fitted Values')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residual Value')

    sm.qqplot(ff3_model.resid, line='45', ax=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')

    axes[1, 1].hist(ff3_model.resid, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].set_xlabel('Residual Value')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_rolling_betas_chart(rolling_betas_df: pd.DataFrame, stock_ticker: str, window_size: int):
    """
    Plots the rolling Fama-French 3-factor betas for a given stock.
    Includes annotations for market events.
    """
    plt.figure(figsize=(14, 7))
    rolling_betas_df.plot(ax=plt.gca())

    plt.title(
        f'Rolling {window_size}-Month Fama-French Factor Betas for {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Beta Value')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(1, color='gray', linestyle='--', alpha=0.5)

    if not rolling_betas_df.empty:
        # Convert to date for comparison
        min_date = rolling_betas_df.index.min().to_pydatetime().date()
        # Convert to date for comparison
        max_date = rolling_betas_df.index.max().to_pydatetime().date()

        # Ensure the annotation dates are within the actual plot's date range
        if min_date <= datetime.date(2020, 2, 1) and max_date >= datetime.date(2020, 4, 1):
            plt.axvspan(datetime.datetime(2020, 2, 1), datetime.datetime(2020, 4, 1),
                        color='red', alpha=0.2, label='COVID-19 Crash')
        if min_date <= datetime.date(2022, 1, 1) and max_date >= datetime.date(2022, 12, 1):
            plt.axvspan(datetime.datetime(2022, 1, 1), datetime.datetime(2022, 12, 1),
                        color='purple', alpha=0.1, label='Inflation/Rate Hikes')

    plt.legend(title='Factor', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_cumulative_return_decomposition(df: pd.DataFrame, ff3_model_params: pd.Series, stock_ticker: str):
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

    market_contribution = beta_M * X['Mkt_RF']
    smb_contribution = beta_S * X['SMB']
    hml_contribution = beta_H * X['HML']
    alpha_contribution = pd.Series(alpha, index=df.index)

    total_model_return = market_contribution + \
        smb_contribution + hml_contribution + alpha_contribution
    epsilon_contribution = y - total_model_return

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
        kind='line', ax=plt.gca(), alpha=0.7, linewidth=2
    )
    attribution_df['Actual Excess Return'].plot(ax=plt.gca(
    ), color='black', linestyle='--', linewidth=2, label='Actual Excess Return')

    plt.title(
        f'Cumulative Return Decomposition for {stock_ticker} (Fama-French 3-Factor Model)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_predicted_vs_actual(df: pd.DataFrame, ff3_model: sm.regression.linear_model.RegressionResultsWrapper, stock_ticker: str):
    """
    Plots predicted vs. actual excess returns with a 45-degree line.

    Args:
        df (pd.DataFrame): Merged DataFrame with stock excess returns and FF factors.
        ff3_model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted FF3 model.
        stock_ticker (str): The ticker symbol of the stock to analyze.
    """
    y_actual = df[f'{stock_ticker}_excess']
    X_predict = df[['Mkt_RF', 'SMB', 'HML']]
    y_predicted = ff3_model.predict(sm.add_constant(X_predict))

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

# --- Summary Table Helper Functions ---


def create_comparative_summary_table(tickers: list, capm_results: dict, ff3_results: dict) -> pd.DataFrame:
    """
    Generates a DataFrame summarizing CAPM vs FF3 regression results.
    """
    summary_data = []
    for stock in tickers:
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
    return pd.DataFrame(summary_data).set_index('Stock')


def create_scenario_summary_table(scenario_projections: dict) -> pd.DataFrame:
    """
    Combines scenario projection results for all stocks into a single DataFrame.
    """
    all_projections_df = pd.DataFrame()
    for stock, projections in scenario_projections.items():
        projections_indexed = projections.set_index('Scenario').rename(
            columns={'Projected_Annual_Excess_Return': stock})
        if all_projections_df.empty:
            all_projections_df = projections_indexed
        else:
            all_projections_df = all_projections_df.join(projections_indexed)
    return all_projections_df.applymap(lambda x: f"{x:.2%}")


def create_final_report_table(tickers: list, capm_results: dict, ff3_results: dict, diagnostic_results: dict) -> pd.DataFrame:
    """
    Generates a comprehensive final report table combining all analysis results.
    """
    final_summary_data = []
    for stock in tickers:
        capm_r = capm_results[stock]
        ff3_r = ff3_results[stock]
        diag_r = diagnostic_results[stock]

        r_squared_improvement = ff3_r['r_squared'] - capm_r['r_squared']
        vifs_str = ", ".join(
            [f"{k}: {v:.2f}" for k, v in diag_r['vif_results'].items() if not np.isnan(v)])

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
    df_final_report_formatted = df_final_report.copy()
    df_final_report_formatted['FF3_Alpha_Ann (%)'] = df_final_report_formatted['FF3_Alpha_Ann (%)'].map(
        '{:.2f}%'.format)
    df_final_report_formatted['FF3_Alpha_pvalue'] = df_final_report_formatted['FF3_Alpha_pvalue'].map(
        '{:.3f}'.format)
    df_final_report_formatted['FF3_Beta_M'] = df_final_report_formatted['FF3_Beta_M'].map(
        '{:.3f}'.format)
    df_final_report_formatted['FF3_Beta_S'] = df_final_report_formatted['FF3_Beta_S'].map(
        '{:.3f}'.format)
    df_final_report_formatted['FF3_Beta_H'] = df_final_report_formatted['FF3_Beta_H'].map(
        '{:.3f}'.format)
    df_final_report_formatted['FF3_R_squared'] = df_final_report_formatted['FF3_R_squared'].map(
        '{:.3f}'.format)
    df_final_report_formatted['R2_Improvement (FF3-CAPM)'] = df_final_report_formatted['R2_Improvement (FF3-CAPM)'].map(
        '{:.3f}'.format)
    df_final_report_formatted['FF3_IR'] = df_final_report_formatted['FF3_IR'].map(
        '{:.3f}'.format)
    df_final_report_formatted['DW_Stat'] = df_final_report_formatted['DW_Stat'].map(
        '{:.3f}'.format)
    df_final_report_formatted['BP_Pvalue'] = df_final_report_formatted['BP_Pvalue'].map(
        '{:.4f}'.format)
    return df_final_report_formatted


# --- Main Orchestration Function ---

def run_factor_analysis(tickers: list, start_date: str, end_date: str, rolling_window: int = 36, macro_scenarios: dict = None, show_plots: bool = True) -> dict:
    """
    Orchestrates the entire factor analysis process, from data retrieval to regressions,
    diagnostics, rolling betas, scenario analysis, and visualizations.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date for data retrieval (e.g., '2014-01-01').
        end_date (str): End date for data retrieval (e.g., '2024-01-01').
        rolling_window (int): The size of the rolling window in months for rolling betas.
        macro_scenarios (dict, optional): A dictionary of macroeconomic scenarios
                                          for return projection. If None, scenario analysis is skipped.
                                          Default is None.
        show_plots (bool): If True, matplotlib plots will be displayed using plt.show().
                           If False, plots will be generated but not displayed.
                           For an app, you might want to replace plt.show() with saving figures
                           to BytesIO or files for dynamic display. Default is True.

    Returns:
        dict: A dictionary containing all computed results and summary DataFrames.
              Includes:
              - 'df_merged': The merged DataFrame with excess returns and factors.
              - 'capm_results': Dictionary of CAPM regression results for each stock.
              - 'ff3_results': Dictionary of Fama-French 3-factor regression results for each stock.
              - 'df_capm_ff3_summary': Comparative summary table of CAPM vs FF3.
              - 'diagnostic_results': Dictionary of diagnostic test results for FF3 models.
              - 'scenario_projections': Dictionary of scenario projection DataFrames (if macro_scenarios provided).
              - 'df_all_projections': Consolidated scenario projections table (if macro_scenarios provided).
              - 'df_final_report': Comprehensive final report table.
    """
    print("--- Starting Factor Analysis ---")

    results = {}

    # 1. Data Retrieval and Merging
    df_merged = retrieve_and_merge_data(tickers, start_date, end_date)
    results['df_merged'] = df_merged
    print("\nFirst 5 rows of the merged data:")
    print(df_merged.head())
    if show_plots and not df_merged.empty:
        # Plot a sample for the first ticker
        plot_data_alignment(df_merged, tickers[0])

    # 2. Run CAPM Regression for all stocks
    capm_results = {}
    print("\n--- Running CAPM Regressions ---")
    for stock in tickers:
        print(f"\n--- CAPM Regression for {stock} ---")
        res = run_capm_regression(df_merged, stock)
        capm_results[stock] = res
        print(res['model'].summary())
        print(f"\nCAPM Results for {stock}:")
        print(
            f"  Alpha (monthly): {res['alpha']:.4f} ({res['alpha_ann']:.2%} annualized)")
        print(f"  Alpha p-value: {res['alpha_pval']:.4f}")
        print(f"  Beta (Market): {res['beta_M']:.3f}")
        print(f"  Beta p-value: {res['beta_M_pval']:.4f}")
        print(f"  R-squared: {res['r_squared']:.3f}")
        print(
            f"  Annualized Residual Std Dev (Tracking Error): {res['resid_std_ann']:.3f}")
        print(f"  Information Ratio: {res['information_ratio']:.3f}")
    results['capm_results'] = capm_results

    # 3. Run Fama-French 3-Factor Regression for all stocks
    ff3_results = {}
    print("\n--- Running Fama-French 3-Factor Regressions ---")
    for stock in tickers:
        print(f"\n--- Fama-French 3-Factor Regression for {stock} ---")
        res = run_ff3_regression(df_merged, stock)
        ff3_results[stock] = res
        print(res['model'].summary())
    results['ff3_results'] = ff3_results

    # 4. Comparative Analysis Table (CAPM vs FF3)
    print("\n--- Comparative Factor Exposure & Performance Table (CAPM vs. FF3) ---")
    df_capm_ff3_summary = create_comparative_summary_table(
        tickers, capm_results, ff3_results)
    print(df_capm_ff3_summary)
    results['df_capm_ff3_summary'] = df_capm_ff3_summary

    # 5. Factor Beta Comparison Bar Chart
    if show_plots:
        plot_factor_betas_comparison(tickers, ff3_results)

    # 6. Security Market Line (SML) Plot
    if show_plots and not df_merged.empty:
        plot_sml_analysis(df_merged, tickers, ff3_results)

    # 7. Run Diagnostic Tests and Plot Diagnostics
    diagnostic_results = {}
    print("\n--- Running Diagnostic Tests for Fama-French 3-Factor Models ---")
    for stock in tickers:
        print(
            f"\n--- Diagnostic Tests for {stock}'s Fama-French 3-Factor Model ---")
        ff3_model = ff3_results[stock]['model']
        X_ff3 = sm.add_constant(df_merged[['Mkt_RF', 'SMB', 'HML']])
        # Removed stock_ticker from args
        res = run_diagnostic_tests(ff3_model, X_ff3)
        diagnostic_results[stock] = res
        print(
            f"  Durbin-Watson statistic: {res['dw_stat']:.3f} - {res['dw_interpretation']}")
        print(
            f"  Breusch-Pagan p-value: {res['bp_pvalue']:.4f} - {res['bp_interpretation']}")
        print(f"  VIF results: {res['vif_results']}")
        print(f"  VIF interpretation: {res['vif_interpretation']}")
        if show_plots:
            plot_regression_diagnostics(ff3_model, stock)
    results['diagnostic_results'] = diagnostic_results

    # 8. Calculate and Plot Rolling Betas
    print(
        f"\n--- Calculating Rolling Betas (Window: {rolling_window} months) ---")
    for stock in tickers:
        print(f"\n--- Rolling Betas for {stock} ---")
        rolling_betas_df = calculate_rolling_betas(
            df_merged, stock, rolling_window)
        if show_plots:
            plot_rolling_betas_chart(rolling_betas_df, stock, rolling_window)

    # 9. Scenario Projections
    if macro_scenarios:
        scenario_projections = {}
        print("\n--- Running Scenario Projections ---")
        for stock in tickers:
            print(f"\n--- Scenario Projections for {stock} ---")
            ff3_model_params = ff3_results[stock]['model'].params
            projections = project_returns_under_scenarios(
                ff3_model_params, macro_scenarios)
            scenario_projections[stock] = projections
            print(f"Annualized Projected Excess Returns for {stock}:")
            print(projections.set_index(
                'Scenario').applymap(lambda x: f"{x:.2%}"))
        results['scenario_projections'] = scenario_projections

        print("\n--- Summary of Projected Annualized Excess Returns Across All Stocks ---")
        df_all_projections = create_scenario_summary_table(
            scenario_projections)
        print(df_all_projections)
        results['df_all_projections'] = df_all_projections
    else:
        print("\n--- Skipping Scenario Projections (no macro_scenarios provided) ---")
        results['scenario_projections'] = None
        results['df_all_projections'] = None

    # 10. Visualizations (Cumulative Return Decomposition & Predicted vs. Actual)
    print("\n--- Generating Performance Visualizations ---")
    for stock in tickers:
        ff3_model = ff3_results[stock]['model']
        ff3_model_params = ff3_model.params
        print(f"\n--- Visualizing Performance for {stock} ---")
        if show_plots:
            plot_cumulative_return_decomposition(
                df_merged, ff3_model_params, stock)
            plot_predicted_vs_actual(df_merged, ff3_model, stock)

    # 11. Final Comparative Summary Table
    print("\n--- Comprehensive Factor Exposure & Performance Report ---")
    df_final_report_formatted = create_final_report_table(
        tickers, capm_results, ff3_results, diagnostic_results)
    print(df_final_report_formatted.to_markdown())
    results['df_final_report'] = df_final_report_formatted

    print("\n--- Factor Analysis Complete ---")

    # Close all plots to free memory. For an app, this might be handled differently.
    if show_plots:
        plt.close('all')

    return results


# --- Example Usage ---
if __name__ == '__main__':
    # Define target stocks and date range
    target_stocks = ['AAPL', 'BRK-B', 'TSLA', 'JNJ']
    analysis_start_date = '2014-01-01'
    analysis_end_date = '2024-01-01'
    rolling_window_size = 36  # 36-month rolling window

    # Define hypothetical macroeconomic scenarios (monthly expected factor returns)
    macro_scenarios = {
        'Base Case': {'Mkt_RF': 0.08/12, 'SMB': 0.01/12, 'HML': 0.005/12},
        'Market Crash': {'Mkt_RF': -0.15/12, 'SMB': -0.05/12, 'HML': 0.02/12},
        'Value Rotation': {'Mkt_RF': 0.01/12, 'SMB': -0.01/12, 'HML': 0.08/12},
        'Small-Cap Rally': {'Mkt_RF': 0.03/12, 'SMB': 0.06/12, 'HML': 0.00/12},
        'Stagflation': {'Mkt_RF': -0.05/12, 'SMB': -0.02/12, 'HML': 0.05/12}
    }

    # Run the full analysis
    print("--- Running example factor analysis ---")
    analysis_results = run_factor_analysis(
        tickers=target_stocks,
        start_date=analysis_start_date,
        end_date=analysis_end_date,
        rolling_window=rolling_window_size,
        macro_scenarios=macro_scenarios,
        show_plots=True  # Set to False if running in a headless environment or if app handles plots
    )

    # You can now access all results via the 'analysis_results' dictionary:
    print("\n--- Accessing results after main function call (sample) ---")
    print("Merged Data Head:\n", analysis_results['df_merged'].head())
    print("\nCAPM Results for AAPL:\n",
          analysis_results['capm_results']['AAPL'])
    print("\nFF3 Results for AAPL:\n", analysis_results['ff3_results']['AAPL'])
    print("\nFinal Report (Markdown):\n",
          analysis_results['df_final_report'].to_markdown())
