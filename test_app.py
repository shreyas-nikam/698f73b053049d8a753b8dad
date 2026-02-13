
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.testing.v1 import AppTest
from unittest import mock
import sys

# --- Dummy implementations of functions from 'source.py' for testing ---
# These functions are mocked to return predictable data, allowing the app logic to proceed.

def retrieve_and_merge_data(tickers, start_date, end_date):
    """Dummy function to simulate data retrieval and merging."""
    # Ensure start_date and end_date are datetime objects
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

    # Calculate approximate number of months
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    if num_months <= 0:
        num_months = 1 # Ensure at least one data point for consistency

    dates = pd.date_range(start=start_date, periods=num_months, freq='M')
    data = {'Mkt_RF': np.random.rand(len(dates)) * 0.05 - 0.02,
            'SMB': np.random.rand(len(dates)) * 0.03 - 0.01,
            'HML': np.random.rand(len(dates)) * 0.03 - 0.01}
    for t in tickers:
        data[f'{t}_excess'] = np.random.rand(len(dates)) * 0.1 - 0.05
    df = pd.DataFrame(data, index=dates)
    return df

class MockOLSResults:
    """A mock statsmodels OLSResults object with necessary attributes."""
    def __init__(self, params, pvalues, rsquared, rsquared_adj=None, n_obs=100, n_params=4):
        self.params = params
        self.pvalues = pvalues
        self.rsquared = rsquared
        self.rsquared_adj = rsquared_adj if rsquared_adj is not None else rsquared - 0.05
        self.resid = pd.Series(np.random.randn(n_obs) * 0.01, index=pd.date_range(start='2014-01-31', periods=n_obs, freq='M'))
        self.summary = mock.MagicMock(return_value="Dummy Regression Summary")
        self.nobs = n_obs
        self.df_model = n_params - 1 # degrees of freedom of the model

    @property
    def fittedvalues(self):
        return pd.Series(np.random.randn(self.nobs) * 0.01, index=pd.date_range(start='2014-01-31', periods=self.nobs, freq='M'))

def run_capm_regression(df_merged, stock_ticker):
    """Dummy CAPM regression result."""
    alpha = np.random.rand() * 0.005 # monthly alpha
    beta_M = np.random.rand() * 0.5 + 0.8 # beta between 0.8 and 1.3
    r_squared = np.random.rand() * 0.6 + 0.2 # r_squared between 0.2 and 0.8
    
    params = {'const': alpha, 'Mkt_RF': beta_M}
    pvalues = {'const': 0.01, 'Mkt_RF': 0.001} # Simulating significance

    mock_model = MockOLSResults(params, pvalues, r_squared, n_params=2) # const + Mkt_RF

    return {
        'model': mock_model,
        'alpha_ann': alpha * 12,
        'alpha_pval': mock_model.pvalues['const'],
        'beta_M': beta_M,
        'beta_M_pval': mock_model.pvalues['Mkt_RF'],
        'r_squared': r_squared,
        'information_ratio': alpha * 12 / (mock_model.resid.std() * np.sqrt(12)) if mock_model.resid.std() > 0 else 0
    }

def run_ff3_regression(df_merged, stock_ticker):
    """Dummy Fama-French 3-Factor regression result."""
    alpha = np.random.rand() * 0.003
    beta_M = np.random.rand() * 0.5 + 0.8
    beta_S = np.random.rand() * 0.5 - 0.25
    beta_H = np.random.rand() * 0.5 - 0.25
    r_squared = np.random.rand() * 0.7 + 0.2
    adj_r_squared = r_squared - 0.05 # simplistic relation
    
    params = {'const': alpha, 'Mkt_RF': beta_M, 'SMB': beta_S, 'HML': beta_H}
    pvalues = {'const': 0.01, 'Mkt_RF': 0.001, 'SMB': 0.05, 'HML': 0.02}

    mock_model = MockOLSResults(params, pvalues, r_squared, adj_r_squared, n_params=4) # const + 3 factors

    return {
        'model': mock_model,
        'alpha_ann': alpha * 12,
        'alpha_pval': mock_model.pvalues['const'],
        'beta_M': beta_M,
        'beta_S': beta_S,
        'beta_H': beta_H,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'information_ratio': alpha * 12 / (mock_model.resid.std() * np.sqrt(12)) if mock_model.resid.std() > 0 else 0
    }

def run_diagnostic_tests(ff3_model, X_ff3_for_vif, stock_ticker):
    """Dummy diagnostic test results."""
    # Ensure X_ff3_for_vif is a DataFrame with expected columns for VIF calc
    # For testing, we can simulate VIF values directly
    vif_results = {'Mkt_RF': np.random.uniform(1.1, 1.5),
                   'SMB': np.random.uniform(1.1, 1.5),
                   'HML': np.random.uniform(1.1, 1.5)}

    return {
        'dw_stat': 1.95,
        'dw_interpretation': "No significant autocorrelation.",
        'bp_pvalue': 0.01, # Simulating heteroskedasticity
        'bp_interpretation': "Suggests heteroskedasticity.",
        'vif_results': vif_results,
        'vif_interpretation': "No significant multicollinearity."
    }

def calculate_rolling_betas(df_merged, stock_ticker, window_size):
    """Dummy rolling betas calculation."""
    if df_merged.empty or len(df_merged) < window_size:
        return pd.DataFrame() # Return empty if not enough data
    dates = df_merged.index[window_size-1:]
    return pd.DataFrame({
        'Mkt_RF': np.random.rand(len(dates)) * 0.5 + 0.8,
        'SMB': np.random.rand(len(dates)) * 0.3 - 0.15,
        'HML': np.random.rand(len(dates)) * 0.3 - 0.15,
    }, index=dates)

def project_returns_under_scenarios(ff3_model_params, macro_scenarios):
    """Dummy scenario projection."""
    projections_data = []
    # Ensure params are accessible, use .get for safety
    alpha = ff3_model_params.get('const', 0)
    beta_M = ff3_model_params.get('Mkt_RF', 0)
    beta_S = ff3_model_params.get('SMB', 0)
    beta_H = ff3_model_params.get('HML', 0)

    for scenario_name, factors in macro_scenarios.items():
        expected_excess_return_monthly = (
            alpha +
            beta_M * factors['Mkt_RF'] +
            beta_S * factors['SMB'] +
            beta_H * factors['HML']
        )
        projections_data.append({
            'Scenario': scenario_name,
            'Projected_Annual_Excess_Return': expected_excess_return_monthly * 12
        })
    return pd.DataFrame(projections_data)

# Create a mock module for 'source.py' and patch it into sys.modules
# This allows 'app.py' to import from our mock when AppTest.from_file loads it.
mock_source = mock.Mock()
mock_source.retrieve_and_merge_data = retrieve_and_merge_data
mock_source.run_capm_regression = run_capm_regression
mock_source.run_ff3_regression = run_ff3_regression
mock_source.run_diagnostic_tests = run_diagnostic_tests
mock_source.calculate_rolling_betas = calculate_rolling_betas
mock_source.project_returns_under_scenarios = project_returns_under_scenarios

sys.modules['source'] = mock_source

# Define the file name for the Streamlit app
APP_FILE = "app.py"

def setup_data_and_models(at_instance):
    """Helper function to set up session state with data and models for subsequent tests."""
    # Ensure we are on the Intro page first
    at_instance.sidebar.selectbox[0].set_value("Introduction & Data Setup").run()

    # Simulate entering tickers and dates
    at_instance.text_input(label="Enter stock tickers (comma-separated):").set_value("AAPL,BRK-B").run()
    at_instance.date_input(label="Start Date:", value=datetime.date(2014, 1, 1)).set_value(datetime.date(2014, 1, 1)).run()
    at_instance.date_input(label="End Date:", value=datetime.date(2024, 1, 1)).set_value(datetime.date(2024, 1, 1)).run()

    # Click the "Retrieve and Prepare Data" button
    at_instance.button("Retrieve and Prepare Data").click().run()

    # Now navigate to CAPM page and run CAPM
    at_instance.sidebar.selectbox[0].set_value("CAPM Baseline").run()
    at_instance.button("Run CAPM Regression for All Stocks").click().run()

    # Navigate to FF3 page and run FF3
    at_instance.sidebar.selectbox[0].set_value("Fama-French 3-Factor Model").run()
    at_instance.button("Run Fama-French 3-Factor Regression for All Stocks").click().run()

    # Optionally run diagnostics for a stock to populate diagnostic_results
    at_instance.sidebar.selectbox[0].set_value("Model Diagnostics").run()
    at_instance.selectbox(label="Select stock for diagnostic tests:", index=0).set_value("AAPL").run()
    at_instance.button("Run Diagnostics for AAPL").click().run()

    return at_instance

def test_initial_page_and_sidebar_defaults():
    at = AppTest.from_file(APP_FILE).run()
    assert at.session_state["current_page"] == "Introduction & Data Setup"
    assert at.session_state["tickers"] == ['AAPL', 'BRK-B', 'TSLA', 'JNJ']
    assert at.sidebar.selectbox[0].value == "Introduction & Data Setup"
    # The title `st.title="QuLab..."` is an assignment, not a component call, so AppTest won't see it.
    # We can check for prominent markdown or headers if they are used to display it.
    assert "Factor Insights for Portfolio Managers" in at.title[0].value

def test_data_setup_page_inputs():
    at = AppTest.from_file(APP_FILE).run()

    # Test ticker input
    at.text_input(label="Enter stock tickers (comma-separated):").set_value("GOOG, MSFT").run()
    assert at.session_state["tickers"] == ["GOOG", "MSFT"]

    # Test date inputs
    new_start_date = datetime.date(2010, 1, 1)
    new_end_date = datetime.date(2020, 12, 31)
    at.date_input(label="Start Date:", value=at.session_state.start_date).set_value(new_start_date).run()
    at.date_input(label="End Date:", value=at.session_state.end_date).set_value(new_end_date).run()
    assert at.session_state["start_date"] == new_start_date
    assert at.session_state["end_date"] == new_end_date

def test_retrieve_data_success():
    at = AppTest.from_file(APP_FILE).run()

    # Input tickers and dates
    at.text_input(label="Enter stock tickers (comma-separated):").set_value("AAPL,BRK-B").run()
    at.date_input(label="Start Date:", value=datetime.date(2014, 1, 1)).set_value(datetime.date(2014, 1, 1)).run()
    at.date_input(label="End Date:", value=datetime.date(2024, 1, 1)).set_value(datetime.date(2024, 1, 1)).run()

    # Click the "Retrieve and Prepare Data" button
    at.button("Retrieve and Prepare Data").click().run()

    # Verify success message and df_merged in session state
    assert at.success[0].value == "Data retrieved and prepared successfully!"
    assert at.session_state["df_merged"] is not None
    assert isinstance(at.session_state["df_merged"], pd.DataFrame)
    assert not at.session_state["df_merged"].empty
    # Verify the dataframe is displayed
    assert at.dataframe[0].value is not None
    # Verify plot is generated (presence of matplotlib figure)
    assert len(at.pyplot) > 0

def test_retrieve_data_no_tickers():
    at = AppTest.from_file(APP_FILE).run()
    at.text_input(label="Enter stock tickers (comma-separated):").set_value("").run() # Clear tickers
    at.button("Retrieve and Prepare Data").click().run()
    assert at.warning[0].value == "Please enter at least one stock ticker."
    assert at.session_state["df_merged"] is None

def test_retrieve_data_failure():
    # Temporarily patch the mock source function to raise an exception
    original_retrieve_data = mock_source.retrieve_and_merge_data
    mock_source.retrieve_and_merge_data.side_effect = Exception("Simulated data retrieval error")

    at = AppTest.from_file(APP_FILE).run()
    at.text_input(label="Enter stock tickers (comma-separated):").set_value("AAPL").run()
    at.button("Retrieve and Prepare Data").click().run()
    assert at.error[0].value == "Error retrieving or preparing data: Simulated data retrieval error"
    assert at.session_state["df_merged"] is None

    # Reset the mock for other tests
    mock_source.retrieve_and_merge_data = original_retrieve_data

def test_capm_regression_success():
    at = AppTest.from_file(APP_FILE).run()
    at = setup_data_and_models(at) # Setup data and navigate to CAPM

    assert "AAPL" in at.session_state["capm_results"]
    assert "BRK-B" in at.session_state["capm_results"]
    assert at.success[0].value == "CAPM regressions completed!"
    # Check for text summary and interpretation for at least one stock
    assert "CAPM Summary for AAPL" in at.text[0].value
    assert "Annualized Alpha:" in at.markdown[4].value # Check for interpretation markdown for AAPL

def test_capm_regression_no_data():
    at = AppTest.from_file(APP_FILE).run()
    at.sidebar.selectbox[0].set_value("CAPM Baseline").run() # Navigate to CAPM
    at.button("Run CAPM Regression for All Stocks").click().run()
    assert at.warning[0].value == "Please retrieve data on the 'Introduction & Data Setup' page first and ensure tickers are selected."
    assert not at.session_state["capm_results"]

def test_ff3_regression_success():
    at = AppTest.from_file(APP_FILE).run()
    at = setup_data_and_models(at) # Setup data and navigate to FF3

    assert "AAPL" in at.session_state["ff3_results"]
    assert "BRK-B" in at.session_state["ff3_results"]
    assert at.success[0].value == "Fama-French 3-Factor regressions completed!"
    # Check for text summary
    assert "FF3 Summary for AAPL" in at.text[1].value # Index for FF3 summary of AAPL
    # Check for the comparative table and plots
    assert at.dataframe[1].value is not None # Comparative table
    assert len(at.pyplot) >= 2 # Factor Beta Comparison, SML Plot (relative to the last page's plots)

def test_ff3_regression_no_data():
    at = AppTest.from_file(APP_FILE).run()
    at.sidebar.selectbox[0].set_value("Fama-French 3-Factor Model").run() # Navigate to FF3
    at.button("Run Fama-French 3-Factor Regression for All Stocks").click().run()
    assert at.warning[0].value == "Please retrieve data on the 'Introduction & Data Setup' page first and ensure tickers are selected."
    assert not at.session_state["ff3_results"]

def test_model_diagnostics_success():
    at = AppTest.from_file(APP_FILE).run()
    at = setup_data_and_models(at) # Setup data and models, and runs diagnostics for AAPL

    assert "AAPL" in at.session_state["diagnostic_results"]
    assert at.success[0].value == "Diagnostic tests completed!"
    assert "Durbin-Watson statistic:" in at.markdown[8].value # Check for diagnostic results markdown
    assert len(at.pyplot) >= 3 # Existing plots + Diagnostic 4-panel plot

def test_model_diagnostics_no_ff3_results():
    at = AppTest.from_file(APP_FILE).run()
    # Only retrieve data, don't run FF3 regression
    at.text_input(label="Enter stock tickers (comma-separated):").set_value("AAPL").run()
    at.button("Retrieve and Prepare Data").click().run()

    at.sidebar.selectbox[0].set_value("Model Diagnostics").run() # Navigate to Diagnostics
    at.button("Run Diagnostics for AAPL").click().run() # Try to run for a selected (default) stock
    assert at.warning[0].value == "Please run Fama-French 3-Factor regressions on the previous page first."
    assert not at.session_state["diagnostic_results"]

def test_rolling_betas_success():
    at = AppTest.from_file(APP_FILE).run()
    at = setup_data_and_models(at) # Setup data and models

    at.sidebar.selectbox[0].set_value("Rolling Betas").run() # Navigate to Rolling Betas
    at.selectbox(label="Select stock for rolling beta analysis:", index=0).set_value("AAPL").run()
    at.slider(label="Select Rolling Window Size (months):", value=at.session_state.rolling_window_size).set_value(48).run()

    at.button("Calculate & Plot Rolling Betas for AAPL").click().run()
    assert at.success[0].value == "Rolling beta analysis completed!"
    assert len(at.pyplot) >= 4 # Existing plots + Rolling Betas plot

def test_rolling_betas_no_data():
    at = AppTest.from_file(APP_FILE).run()
    at.sidebar.selectbox[0].set_value("Rolling Betas").run()
    # The default selected stock is 'AAPL'
    at.button("Calculate & Plot Rolling Betas for AAPL").click().run()
    assert at.warning[0].value == "Please retrieve data on the 'Introduction & Data Setup' page first and ensure tickers are selected."

def test_scenario_analysis_modify_and_add_scenario():
    at = AppTest.from_file(APP_FILE).run()
    at = setup_data_and_models(at) # Setup data and models

    at.sidebar.selectbox[0].set_value("Scenario Analysis").run() # Navigate to Scenario Analysis

    # Modify an existing scenario
    at.number_input(key="Base Case_Mkt_RF").set_value(0.01).run()
    assert at.session_state.macro_scenarios["Base Case"]["Mkt_RF"] == 0.01

    # Add a new scenario
    at.expander[0].open().text_input(key="new_scenario_name").set_value("New Boom").run()
    at.expander[0].number_input(key="new_mkt").set_value(0.02).run()
    at.expander[0].number_input(key="new_smb").set_value(0.01).run()
    at.expander[0].number_input(key="new_hml").set_value(0.005).run()
    at.expander[0].button("Add Scenario").click().run()

    # Rerun to pick up the rerender and verify the success message and updated state
    assert "New Boom" in at.session_state.macro_scenarios
    assert at.session_state.macro_scenarios["New Boom"]["Mkt_RF"] == 0.02
    assert at.success[0].value == "Scenario 'New Boom' added!"


def test_scenario_analysis_delete_scenario():
    at = AppTest.from_file(APP_FILE).run()
    at = setup_data_and_models(at) # Setup data and models

    at.sidebar.selectbox[0].set_value("Scenario Analysis").run() # Navigate to Scenario Analysis

    # Delete an existing scenario (e.g., 'Market Crash')
    at.button(key="delete_Market Crash").click().run() # This click triggers st.rerun internally
    
    assert "Market Crash" not in at.session_state.macro_scenarios
    assert at.success[0].value == "Scenario(s) deleted. Please re-run projection."

def test_scenario_analysis_projection_success():
    at = AppTest.from_file(APP_FILE).run()
    at = setup_data_and_models(at) # Setup data and models

    at.sidebar.selectbox[0].set_value("Scenario Analysis").run() # Navigate to Scenario Analysis
    at.button("Project Returns Under Scenarios").click().run()

    assert at.success[0].value == "Scenario projections completed!"
    assert "AAPL" in at.session_state["scenario_projections"]
    assert "BRK-B" in at.session_state["scenario_projections"]
    assert not at.session_state["scenario_projections"]["AAPL"].empty
    assert at.dataframe[2].value is not None # Summary table of projections

def test_scenario_analysis_no_ff3_results():
    at = AppTest.from_file(APP_FILE).run()
    # Only retrieve data, don't run FF3 regression
    at.text_input(label="Enter stock tickers (comma-separated):").set_value("AAPL").run()
    at.button("Retrieve and Prepare Data").click().run()

    at.sidebar.selectbox[0].set_value("Scenario Analysis").run() # Navigate to Scenario Analysis
    at.button("Project Returns Under Scenarios").click().run()
    assert at.warning[0].value == "Please run Fama-French 3-Factor regressions on the 'Fama-French 3-Factor Model' page first and ensure tickers are selected."
    assert not at.session_state["scenario_projections"]

def test_performance_attribution_success():
    at = AppTest.from_file(APP_FILE).run()
    at = setup_data_and_models(at) # Setup data and models

    at.sidebar.selectbox[0].set_value("Performance Attribution & Report").run() # Navigate
    at.selectbox(label="Select stock for performance attribution:", index=0).set_value("AAPL").run()

    at.button("Generate Performance Attribution for AAPL").click().run()
    assert at.success[0].value == "Performance attribution plots generated!"
    assert len(at.pyplot) >= 5 # Existing plots + 2 attribution plots

def test_performance_attribution_no_ff3_results():
    at = AppTest.from_file(APP_FILE).run()
    # Only retrieve data, don't run FF3 regression
    at.text_input(label="Enter stock tickers (comma-separated):").set_value("AAPL").run()
    at.button("Retrieve and Prepare Data").click().run()

    at.sidebar.selectbox[0].set_value("Performance Attribution & Report").run() # Navigate
    at.button("Generate Performance Attribution for AAPL").click().run()
    assert at.warning[0].value == "Please run Fama-French 3-Factor regressions on the 'Fama-French 3-Factor Model' page first and ensure tickers are selected."

def test_performance_attribution_full_report():
    at = AppTest.from_file(APP_FILE).run()
    at = setup_data_and_models(at) # Setup data and models, runs diagnostics too.

    # Ensure all required session states are populated for the report to render
    at.sidebar.selectbox[0].set_value("Performance Attribution & Report").run()
    # The report renders automatically if session states are ready, no button click needed
    assert at.dataframe[2].value is not None # Final comprehensive report
    assert "FF3_Alpha_Ann (%)" in at.dataframe[2].value.columns
    assert "VIFs" in at.dataframe[2].value.columns

