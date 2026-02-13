# QuLab: Lab 4 - Factor Insights for Portfolio Managers (Predicting Stock Beta and Beyond)

## 📊 Project Overview

This Streamlit application, "QuLab: Lab 4: Factor Insights for Portfolio Managers," is an interactive tool designed for financial professionals, particularly portfolio managers, to delve into systematic risk analysis and return attribution using classical asset pricing models. It moves beyond simple market beta to explore multi-factor models, providing a robust framework for performance attribution, risk management, and scenario-based return forecasting.

The application guides users through a structured workflow, from data acquisition and preparation to running and diagnosing regression models, analyzing dynamic factor exposures, projecting returns under various scenarios, and finally generating comprehensive performance reports. This project aims to demonstrate how Python and Streamlit can be leveraged to build reproducible and insightful financial analysis tools, replacing traditional spreadsheet-based, error-prone processes.

**Persona:** The application is built with the persona of "Alex, a CFA Charterholder and Portfolio Manager at 'Alpha Investments'," in mind, illustrating his journey from basic market beta analysis to sophisticated multi-factor modeling.

## ✨ Key Features

The application provides a comprehensive suite of tools for factor analysis:

1.  **Introduction & Data Setup**:
    *   **Interactive Data Acquisition**: Fetch historical monthly total returns for user-defined stock tickers and Fama-French factor data (Mkt-RF, SMB, HML, RF).
    *   **Flexible Date Range**: Specify custom start and end dates for analysis.
    *   **Data Alignment Validation**: Visual plot to confirm correct alignment of stock excess returns with market excess returns.

2.  **CAPM Baseline Analysis**:
    *   **Capital Asset Pricing Model (CAPM) Regression**: Estimate market beta ($\beta_M$) and Jensen's Alpha for selected stocks.
    *   **Detailed Regression Output**: Display `statsmodels` summary tables for in-depth statistical review.
    *   **Key Metrics**: Calculate and present annualized Alpha, Market Beta, R-squared, and Information Ratio.

3.  **Fama-French 3-Factor Model**:
    *   **Multi-Factor Regression**: Run the Fama-French 3-Factor Model, including market (Mkt-RF), size (SMB), and value (HML) factors.
    *   **Comprehensive Factor Exposure**: Quantify exposure to market, size, and value factors ($\beta_M, \beta_S, \beta_H$).
    *   **Comparative Analysis**: Side-by-side comparison of CAPM and FF3 results, highlighting R-squared improvement and Information Ratio.
    *   **Visualizations**: Bar charts for factor beta comparison across stocks and a Security Market Line (SML) plot to visualize risk-return relationships.

4.  **Model Diagnostics**:
    *   **Robustness Checks**: Perform diagnostic tests to validate OLS regression assumptions for the Fama-French 3-Factor Model.
    *   **Tests Included**:
        *   **Durbin-Watson statistic**: Detects autocorrelation in residuals.
        *   **Breusch-Pagan test**: Checks for heteroskedasticity.
        *   **Variance Inflation Factor (VIF)**: Identifies multicollinearity among factors.
    *   **4-Panel Residual Plots**: Visualize residuals over time, against fitted values, Q-Q plot for normality, and a histogram of residuals.
    *   **Practitioner Warnings**: Emphasizes the importance of HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors in financial time series.

5.  **Rolling Betas**:
    *   **Dynamic Risk Analysis**: Calculate and visualize rolling market, size, and value betas over a user-defined window (e.g., 36 months).
    *   **Time-Varying Factor Exposures**: Understand how a stock's sensitivity to factors changes across different market regimes and economic events.
    *   **Event Annotation**: Plots may highlight major market events like the COVID-19 crash or periods of inflation.

6.  **Scenario Analysis**:
    *   **Forward-Looking Projections**: Project expected annualized excess returns for stocks under various hypothetical macroeconomic scenarios.
    *   **Customizable Scenarios**: Define and modify expected monthly returns for Mkt-RF, SMB, and HML for different scenarios (e.g., "Market Crash", "Value Rotation", "Stagflation").
    *   **Strategic Planning Tool**: Aids in stress testing portfolios and making tactical asset allocation decisions.

7.  **Performance Attribution & Report**:
    *   **Cumulative Return Decomposition**: Visualize the contribution of each factor (Market, SMB, HML), Alpha, and Residual to a stock's cumulative excess return.
    *   **Predicted vs. Actual Plot**: Scatter plot comparing model-predicted excess returns against actual excess returns for a visual fit assessment.
    *   **Comprehensive Summary Report**: A consolidated table summarizing key FF3 metrics, R-squared improvement, Information Ratio, and diagnostic test results for all analyzed stocks, suitable for reporting.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/qu-lab4-factor-insights.git
    cd qu-lab4-factor-insights
    ```
    (Replace `your-username/qu-lab4-factor-insights.git` with the actual repository URL if this is hosted publicly.)

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages**:
    Create a `requirements.txt` file in the project root with the following content:

    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    statsmodels
    yfinance
    requests
    scipy
    ```

    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure `source.py` is present**:
    Make sure the `source.py` file (containing the backend functions like `retrieve_and_merge_data`, `run_capm_regression`, etc.) is in the same directory as your main Streamlit application file (e.g., `app.py`).

## 💻 Usage

1.  **Run the Streamlit application**:
    From your project directory, with your virtual environment activated, run:
    ```bash
    streamlit run app.py
    ```
    (Assuming your main application file is named `app.py`. If it's `main.py` or similar, adjust the command.)

2.  **Access the Application**:
    Your web browser should automatically open to the application's local URL (usually `http://localhost:8501`).

3.  **Navigate and Interact**:
    *   Use the **"Factor Insights Navigator"** sidebar to switch between different analysis pages.
    *   **"Introduction & Data Setup"**: Enter stock tickers (e.g., `AAPL, BRK-B, TSLA, JNJ`), adjust dates, and click "Retrieve and Prepare Data" to load historical stock and factor data. This step is crucial before proceeding.
    *   **Subsequent Pages**: Follow the instructions on each page to run regressions, perform diagnostics, visualize rolling betas, define scenarios, and generate reports. Interact with input widgets (text inputs, sliders, buttons) to control the analysis.

## 📁 Project Structure

```
qu-lab4-factor-insights/
├── app.py                  # Main Streamlit application file
├── source.py               # Backend functions for data retrieval, regressions, diagnostics, etc.
├── requirements.txt        # List of Python dependencies
└── README.md               # This README file
```

## 🛠 Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/) - For building interactive web applications with Python.
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/) - For data structures and analysis.
*   **Numerical Operations**: [NumPy](https://numpy.org/) - For numerical computing.
*   **Statistical Modeling**: [Statsmodels](https://www.statsmodels.org/stable/index.html) - For estimating statistical models and performing diagnostic tests.
*   **Plotting & Visualization**:
    *   [Matplotlib](https://matplotlib.org/) - For creating static, animated, and interactive visualizations.
    *   [Seaborn](https://seaborn.pydata.org/) - For statistical data visualization based on Matplotlib.
*   **Data Retrieval (Implied by `source.py`)**:
    *   [yfinance](https://pypi.org/project/yfinance/) - For fetching historical stock data from Yahoo Finance.
    *   [requests](https://docs.python-requests.org/en/latest/) - For making HTTP requests, likely used to fetch Fama-French factor data from external sources (e.g., Ken French's data library).

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and commit them (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
(If you don't have a `LICENSE` file, you should create one or state "No License".)

## ✉️ Contact

For any questions or feedback, please reach out to:

*   **QuantUniversity** - [https://www.quantuniversity.com/](https://www.quantuniversity.com/)
*   **Project Maintainer**: [Your Name/Email/LinkedIn] (Optional)


## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
