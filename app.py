import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Import only what we need from source.py to avoid namespace clashes.
from source import (
    retrieve_and_merge_data,
    run_capm_regression,
    run_ff3_regression,
    run_diagnostic_tests,
    calculate_rolling_betas,
    project_returns_under_scenarios
)

# --- Page Configuration ---
st.set_page_config(
    page_title="QuLab: Lab 4: Predicting Stock Beta (Regression)",
    layout="wide"
)

# Sidebar brand + navigation
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()

st.title("QuLab: Lab 4: Predicting Stock Beta (Regression)")
st.divider()

# Visual defaults
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("deep")

# -----------------------------
# Utilities (learning-first)
# -----------------------------


def _has_data_ready() -> bool:
    return (
        st.session_state.df_merged is not None
        and isinstance(st.session_state.df_merged, pd.DataFrame)
        and not st.session_state.df_merged.empty
        and bool(st.session_state.tickers)
    )


def _fmt_pct(x, decimals=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{decimals}%}"


def _fmt_num(x, decimals=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{decimals}f}"


def _pval_badge(p):
    # Simple, CFA-friendly significance cue without overexplaining
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=—"
    if p < 0.01:
        return f"p={p:.4f} (strong)"
    if p < 0.05:
        return f"p={p:.4f} (moderate)"
    return f"p={p:.4f} (weak)"


def _checkpoint(question: str, options, key: str, help_text: str = ""):
    st.markdown("**Checkpoint (quick self-test):**")
    # Using index=None to prevent immediate answer reveal (requires Streamlit 1.20+)
    selected = st.radio(
        question, options, key=f"radio_{key}", help=None, horizontal=False, index=None)
    if selected:
        # For this lab, assume the first option is consistently the correct answer
        if options.index(selected) == 0:
            st.success(f"**Correct!** {help_text}")
        else:
            st.error(f"**Not quite.** {help_text}")


def _assumptions_box(title: str, bullets):
    with st.expander(title, expanded=False):
        for b in bullets:
            st.markdown(f"- {b}")


def _decision_translation(lines):
    st.markdown("**Decision translation:**")
    for line in lines:
        st.markdown(f"- {line}")


def _show_state_of_progress():
    # A compact “you are here” checklist for cognitive clarity
    steps = [
        ("Data loaded & aligned", _has_data_ready()),
        ("CAPM estimated", bool(st.session_state.capm_results)),
        ("FF3 estimated", bool(st.session_state.ff3_results)),
        ("Diagnostics run (at least one stock)",
         bool(st.session_state.diagnostic_results)),
        ("Rolling betas computed (at least one stock)", bool(
            st.session_state.get("last_rolling_betas_stock"))),
        ("Scenario projections computed", bool(
            st.session_state.scenario_projections)),
    ]


# -----------------------------
# Session state init
# -----------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Introduction & Data Setup"

if "tickers" not in st.session_state:
    st.session_state.tickers = ["AAPL", "BRK-B", "TSLA", "JNJ"]

if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.date(2014, 1, 1)

if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.date(2024, 1, 1)

if "df_merged" not in st.session_state:
    st.session_state.df_merged = None

if "capm_results" not in st.session_state:
    st.session_state.capm_results = {}

if "ff3_results" not in st.session_state:
    st.session_state.ff3_results = {}

if "diagnostic_results" not in st.session_state:
    st.session_state.diagnostic_results = {}

if "rolling_window_size" not in st.session_state:
    st.session_state.rolling_window_size = 36

if "macro_scenarios" not in st.session_state:
    st.session_state.macro_scenarios = {
        "Base Case": {"Mkt_RF": 0.08 / 12, "SMB": 0.01 / 12, "HML": 0.005 / 12},
        "Market Crash": {"Mkt_RF": -0.15 / 12, "SMB": -0.05 / 12, "HML": 0.02 / 12},
        "Value Rotation": {"Mkt_RF": 0.01 / 12, "SMB": -0.01 / 12, "HML": 0.08 / 12},
        "Small-Cap Rally": {"Mkt_RF": 0.03 / 12, "SMB": 0.06 / 12, "HML": 0.00 / 12},
        "Stagflation": {"Mkt_RF": -0.05 / 12, "SMB": -0.02 / 12, "HML": 0.05 / 12},
    }

if "scenario_projections" not in st.session_state:
    st.session_state.scenario_projections = {}

if "last_rolling_betas_stock" not in st.session_state:
    st.session_state.last_rolling_betas_stock = None

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Factor Insights Navigator")

page_options = [
    "Introduction & Data Setup",
    "CAPM Baseline",
    "Fama-French 3-Factor Model",
    "Model Diagnostics",
    "Rolling Betas",
    "Scenario Analysis",
    "Performance Attribution & Report",
]

current_index = 0
if st.session_state.current_page in page_options:
    current_index = page_options.index(st.session_state.current_page)

st.session_state.current_page = st.sidebar.selectbox(
    "Go to page:",
    page_options,
    index=current_index
)

st.sidebar.divider()
_show_state_of_progress()

with st.sidebar.expander("How to use this lab (60 seconds)", expanded=False):
    st.markdown(
        "- Start on **Introduction & Data Setup** and load data.\n"
        "- Estimate **CAPM** to get a baseline market beta.\n"
        "- Estimate **FF3** to see whether size/value exposures materially change conclusions.\n"
        "- Run **Diagnostics** to judge whether inference (p-values) is trustworthy.\n"
        "- Use **Rolling Betas** to detect regime changes.\n"
        "- Use **Scenario Analysis** to translate betas into forward-looking stress tests.\n"
        "- Use **Performance Attribution & Report** to communicate results to stakeholders."
    )

# -----------------------------
# Page Rendering
# -----------------------------
if st.session_state.current_page == "Introduction & Data Setup":
    st.title("Predicting Beta: A Decision Tool, Not a Spreadsheet Exercise")

    st.markdown(f"")
    st.markdown(f"**Persona:** Alex, a CFA Charterholder and Portfolio Manager at 'Alpha Investments', is transitioning to Python for more robust and scalable financial analysis.")
    st.markdown(f"")
    st.markdown(f"His goal is to move beyond simple market beta to multi-factor models for performance attribution, systematic risk management, and return forecasting.")
    st.markdown(f"")

    st.info(
        "Learning objective: you should leave this page able to explain (i) what an excess return is, "
        "(ii) what is being merged, and (iii) how you verify the merge is *not silently wrong*."
    )

    st.markdown(f"### 1. Data Acquisition and Preparation")
    st.markdown(
        f"Alex needs to gather historical monthly total returns for his target stocks and the widely-used "
        f"Fama-French factor returns, along with the risk-free rate. This process, often manual and error-prone "
        f"in spreadsheets, is automated and made reproducible in Python."
    )
    st.markdown(f"")
    st.markdown(
        f"He'll select a diversified set of stocks to analyze and merge them with the Fama-French data.")
    st.markdown(f"")

    st.markdown(
        f"**Practitioner Warning:** Fama-French factors use end-of-month dates while Yahoo Finance may use different "
        f"conventions. Always verify that the merge is correct by spot-checking known dates (e.g., March 2020 COVID "
        f"crash should show large negative Mkt-RF). A one-month misalignment would produce meaningless regressions. "
        f"Also note that Fama-French returns are in percent (e.g., 2.5 = 2.5%) while Yahoo Finance returns are in "
        f"decimal (0.025). Convert before merging."
    )
    st.markdown(f"")

    _assumptions_box(
        "Assumptions & definitions (click to expand)",
        [
            "All analysis is on **monthly** data.",
            "Returns used in regressions are **excess returns** (asset return minus risk-free rate).",
            "Fama-French factor inputs are interpreted as **factor excess returns** (e.g., Mkt-RF).",
            "If inputs are mis-scaled (percent vs decimal), betas and alphas become meaningless.",
        ],
    )

    st.subheader("Configure Data Retrieval")

    # Define a list of common tickers as options for the multiselect
    ticker_options = sorted(["AAPL", "MSFT", "TSLA", "JNJ", "BRK-B",
                            "AMZN", "GOOGL", "META", "NVDA", "V", "PG", "UNH", "HD", "DIS"])
    # Ensure any current tickers in session state are also in the options list
    all_options = sorted(list(set(ticker_options + st.session_state.tickers)))

    st.session_state.tickers = st.multiselect(
        "Tickers to analyze",
        options=all_options,
        default=st.session_state.tickers,
        help="Select liquid equities for a cleaner learning experience. Examples: AAPL, MSFT, JNJ, BRK-B."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input(
            "Start date (inclusive)",
            value=st.session_state.start_date,
            help="Longer samples stabilize beta but can blur regime shifts."
        )
    with col2:
        st.session_state.end_date = st.date_input(
            "End date (inclusive)",
            value=st.session_state.end_date,
            help="Choose an end date that includes at least one meaningful regime (e.g., 2020, 2022)."
        )

    st.markdown("**Quick sanity checks before you pull data:**")
    _checkpoint(
        "If your sample is only 24 months, what will you worry about most?",
        ["Unstable beta estimates / wide confidence intervals",
            "Overfitting a black-box model", "Dividends not included"],
        key="cp_data_1",
        help_text="Short samples lead to noisy coefficient estimates and fragile inference."
    )

    if st.button("Retrieve and Prepare Data"):
        if st.session_state.tickers:
            with st.spinner("Fetching and merging data..."):
                try:
                    df_merged_temp = retrieve_and_merge_data(
                        st.session_state.tickers,
                        st.session_state.start_date.strftime("%Y-%m-%d"),
                        st.session_state.end_date.strftime("%Y-%m-%d"),
                    )
                    st.session_state.df_merged = df_merged_temp
                    st.success("Data retrieved and prepared successfully!")

                    # Display: show enough to validate schema + scale
                    st.markdown(
                        "**Preview (first rows): validate columns and scale**")
                    st.dataframe(st.session_state.df_merged.head())

                    # Alignment verification plot (kept, but framed as a decision-quality check)
                    if st.session_state.tickers and not st.session_state.df_merged.empty:
                        first_ticker = st.session_state.tickers[0]
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(
                            st.session_state.df_merged.index,
                            st.session_state.df_merged[f"{first_ticker}_excess"],
                            label=f"{first_ticker} Excess Return"
                        )
                        ax.plot(
                            st.session_state.df_merged.index,
                            st.session_state.df_merged["Mkt_RF"],
                            label="Market Excess Return (Mkt-RF)"
                        )
                        ax.set_title(
                            f"Sanity Check: {first_ticker} Excess Return vs. Mkt-RF (Alignment + Scale)")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Monthly Return")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)

                        _decision_translation([
                            "If the plot looks wildly off-scale (e.g., 2.0 instead of 0.02), you likely have a percent/decimal mismatch.",
                            "If known stress windows (e.g., early 2020) don’t show the expected sign/magnitude in Mkt-RF, suspect date misalignment.",
                        ])

                except Exception as e:
                    st.error(f"Error retrieving or preparing data: {e}")
        else:
            st.warning("Please enter at least one stock ticker.")

    if st.session_state.df_merged is not None:
        st.markdown(
            f"**Data readiness:** {st.session_state.df_merged.shape[0]} monthly observations loaded. "
            f"Next: estimate CAPM to obtain a baseline market beta."
        )

elif st.session_state.current_page == "CAPM Baseline":
    st.title("CAPM Baseline: Market Beta as a First-Pass Risk Descriptor")

    st.markdown(
        f"### 2. Establishing a Baseline: The Capital Asset Pricing Model (CAPM)")
    st.markdown(
        f"Before diving into complex multi-factor models, Alex starts with the fundamental Capital Asset Pricing Model (CAPM) "
        f"to establish a baseline understanding of each stock's sensitivity to the overall market. "
        f"The CAPM is a single-factor model that explains the expected return of an asset based on its market risk."
    )

    st.markdown(
        r"""
$$
R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \epsilon_{i,t}
$$
"""
    )
    st.markdown(
        r"where $R_{i,t} - R_{f,t}$ is the excess return of asset $i$ at time $t$.")
    st.markdown(
        r"$\alpha_i$ (Jensen's Alpha) is the asset's abnormal return not explained by the market factor. "
        r"A positive and statistically significant $\alpha_i$ indicates outperformance after adjusting for market risk."
    )
    st.markdown(
        r"$\beta_{i,M}$ (Market Beta) measures the asset's sensitivity to market movements. "
        r"A $\beta_{i,M} > 1$ implies higher market sensitivity than the average stock, while $\beta_{i,M} < 1$ implies lower sensitivity."
    )
    st.markdown(
        r"$R_{m,t} - R_{f,t}$ is the market excess return at time $t$.")
    st.markdown(r"$\epsilon_{i,t}$ is the idiosyncratic error term.")
    st.markdown(f"")
    st.markdown(
        f"Alex will perform this regression for each of his target stocks to obtain their individual market betas and Jensen's alpha, "
        f"along with statistical significance."
    )
    st.markdown(f"")

    _assumptions_box(
        "What you can and cannot claim from CAPM (click to expand)",
        [
            "CAPM beta is a **linear** sensitivity estimate to Mkt-RF over the sample window.",
            "A statistically insignificant alpha does **not** prove “no skill”; it means the sample does not support rejecting alpha=0.",
            "High R-squared means “market explains more of the variation,” not “the stock is safer.”",
        ],
    )

    if not _has_data_ready():
        st.warning(
            "Please retrieve data on the 'Introduction & Data Setup' page first.")
    else:
        colA, colB = st.columns([0.55, 0.45])
        with colA:
            capm_mode = st.radio(
                "Run CAPM for:",
                ["One stock (recommended for learning)",
                 "All stocks (batch run)"],
                help="Start with one stock to internalize how alpha/beta/p-values map to decisions."
            )
        with colB:
            selected_stock_single = st.selectbox(
                "Choose a stock (for interpretation)",
                st.session_state.tickers,
                help="Pick a stock where you already have a prior belief about beta (e.g., TSLA vs JNJ)."
            )

        if capm_mode == "One stock (recommended for learning)":
            if st.button(f"Run CAPM Regression for {selected_stock_single}"):
                with st.spinner(f"Running CAPM for {selected_stock_single}..."):
                    try:
                        results = run_capm_regression(
                            st.session_state.df_merged, selected_stock_single)
                        st.session_state.capm_results[selected_stock_single] = results
                        st.success("CAPM regression completed!")
                    except Exception as e:
                        st.error(
                            f"Error running CAPM for {selected_stock_single}: {e}")

        else:
            if st.button("Run CAPM Regression for All Stocks"):
                st.session_state.capm_results = {}
                for stock in st.session_state.tickers:
                    with st.spinner(f"Running CAPM for {stock}..."):
                        try:
                            results = run_capm_regression(
                                st.session_state.df_merged, stock)
                            st.session_state.capm_results[stock] = results
                        except Exception as e:
                            st.error(f"Error running CAPM for {stock}: {e}")
                st.success("CAPM regressions completed!")

        # Display: interpretation-first, then full stats output
        if selected_stock_single in st.session_state.capm_results:
            r = st.session_state.capm_results[selected_stock_single]

            st.subheader(
                f"CAPM results (interpretation-first): {selected_stock_single}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Market beta (βM)", _fmt_num(r.get("beta_M")))
            m2.metric("Annualized alpha", _fmt_pct(r.get("alpha_ann")))
            m3.metric("R-squared", _fmt_num(r.get("r_squared")))
            m4.metric("Information ratio", _fmt_num(
                r.get("information_ratio")))

            st.markdown(
                f"- **Alpha inference:** {_pval_badge(r.get('alpha_pval'))} (testing whether alpha is distinguishable from 0 in-sample)\n"
                f"- **Beta inference:** {_pval_badge(r.get('beta_M_pval'))} (testing whether beta is distinguishable from 0)"
            )

            _decision_translation([
                "If βM is materially > 1, this holding will amplify portfolio drawdowns in broad market sell-offs unless offset elsewhere.",
                "If βM is materially < 1, this holding can dampen market-driven volatility but may lag in risk-on environments.",
                "If alpha is not statistically meaningful, treat “outperformance” claims as fragile unless supported by out-of-sample evidence or a structural thesis.",
            ])

            with st.expander("Full regression output (for auditability)", expanded=False):
                st.text(r["model"].summary())

            _checkpoint(
                "A stock has βM≈1.3 and R²≈0.60. What does that *most defensibly* mean?",
                [
                    "It tends to move 30% more than the market, and market explains a meaningful share of its variation.",
                    "It is 30% more likely to beat the market next year.",
                    "It is safer because R² is high."
                ],
                key="cp_capm_1",
                help_text="Beta is sensitivity; R² is explanatory share of variance (not a safety metric)."
            )

        elif st.session_state.capm_results:
            st.info(
                "CAPM has been run for at least one stock. Select a stock above to view interpretation.")

elif st.session_state.current_page == "Fama-French 3-Factor Model":
    st.title("Fama-French 3-Factor Model: From ‘Market Only’ to Factor Fingerprints")

    st.markdown(f"### 3. Deeper Insights with Fama-French 3-Factor Model")
    st.markdown(
        f"While CAPM provides a basic understanding, Alex knows that investment performance is often driven by more than just market risk. "
        f"He moves to the Fama-French 3-Factor Model, which adds size (SMB) and value (HML) factors, offering a richer explanation of asset returns "
        f"and a more nuanced performance attribution."
    )

    st.markdown(
        r"""
$$
R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M}(R_{m,t} - R_{f,t}) + \beta_{i,S}SMB_t + \beta_{i,H}HML_t + \epsilon_{i,t}
$$
"""
    )
    st.markdown(
        r"where $\beta_{i,S}$ (Size Beta) measures the asset's exposure to the small-cap factor. A positive $\beta_{i,S}$ suggests a tilt towards smaller companies."
    )
    st.markdown(
        r"where $\beta_{i,H}$ (Value Beta) measures the asset's exposure to the value factor. A positive $\beta_{i,H}$ suggests a tilt towards value stocks (high book-to-market), while a negative $\beta_{i,H}$ indicates a growth stock tilt (low book-to-market)."
    )
    st.markdown(r"where Other terms are as defined in the CAPM.")
    st.markdown(f"")
    st.markdown(
        f"Alex will run this model for all stocks, compare their factor exposures (their \"factor fingerprints\"), "
        f"and quantify the incremental explanatory power of the additional factors (SMB and HML). He'll also use the Information Ratio, defined as:"
    )
    st.markdown(
        r"""
$$
IR = \frac{\hat{\alpha}_{\text{ann}}}{\hat{\sigma}_{\epsilon}\sqrt{12}} = \frac{\text{Annualized Alpha}}{\text{Annualized Tracking Error}}
$$
"""
    )
    st.markdown(r"where The Information Ratio measures the risk-adjusted abnormal return, where $|IR| > 0.5$ is considered strong performance.")
    st.markdown(f"")

    _assumptions_box(
        "Interpretation guardrails for FF3 (click to expand)",
        [
            "SMB and HML betas are **exposures**, not “styles you intentionally run.” Exposures can be incidental.",
            "A higher R-squared vs CAPM supports that additional factors help explain variation, not that the model is ‘true.’",
            "FF3 is still linear and can miss sector shocks, convexity, and event risk.",
        ],
    )

    if not _has_data_ready():
        st.warning(
            "Please retrieve data on the 'Introduction & Data Setup' page first.")
    else:
        ff3_mode = st.radio(
            "Run FF3 for:",
            ["One stock (recommended for learning)", "All stocks (batch run)"],
            help="Start with one stock to build intuition for how SMB/HML shift your interpretation vs CAPM."
        )
        selected_stock_ff3 = st.selectbox(
            "Choose a stock (for interpretation)",
            st.session_state.tickers,
            help="Pick a stock where you suspect a growth tilt (negative HML) or value tilt (positive HML)."
        )

        if ff3_mode == "One stock (recommended for learning)":
            if st.button(f"Run FF3 Regression for {selected_stock_ff3}"):
                with st.spinner(f"Running FF3 for {selected_stock_ff3}..."):
                    try:
                        results = run_ff3_regression(
                            st.session_state.df_merged, selected_stock_ff3)
                        st.session_state.ff3_results[selected_stock_ff3] = results
                        st.success("FF3 regression completed!")
                    except Exception as e:
                        st.error(
                            f"Error running FF3 for {selected_stock_ff3}: {e}")
        else:
            if st.button("Run Fama-French 3-Factor Regression for All Stocks"):
                st.session_state.ff3_results = {}
                for stock in st.session_state.tickers:
                    with st.spinner(f"Running FF3 for {stock}..."):
                        try:
                            results = run_ff3_regression(
                                st.session_state.df_merged, stock)
                            st.session_state.ff3_results[stock] = results
                        except Exception as e:
                            st.error(f"Error running FF3 for {stock}: {e}")
                st.success("Fama-French 3-Factor regressions completed!")

        # Interpretation-first panel for the chosen stock
        if selected_stock_ff3 in st.session_state.ff3_results:
            r = st.session_state.ff3_results[selected_stock_ff3]
            st.subheader(
                f"FF3 results (interpretation-first): {selected_stock_ff3}")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("βM", _fmt_num(r.get("beta_M")))
            c2.metric("βSMB", _fmt_num(r.get("beta_S")))
            c3.metric("βHML", _fmt_num(r.get("beta_H")))
            c4.metric("R²", _fmt_num(r.get("r_squared")))
            c5.metric("Annualized α", _fmt_pct(r.get("alpha_ann")))

            st.markdown(
                f"- **Alpha inference:** {_pval_badge(r.get('alpha_pval'))}")

            _decision_translation([
                "If βHML is meaningfully negative, the stock behaves more like a growth exposure (and may underperform in value rotations).",
                "If βSMB is meaningfully positive, the stock has small-cap sensitivity (often more cyclicality/liquidity risk).",
                "If R² improves materially vs CAPM, factor exposures beyond the market are important for risk attribution and stress testing.",
            ])

            with st.expander("Full regression output (for auditability)", expanded=False):
                st.text(r["model"].summary())

        # Comparative table + plots (kept, but framed as learning outputs)
        if st.session_state.capm_results and st.session_state.ff3_results and st.session_state.tickers:
            st.markdown(
                "### Comparative Factor Exposure & Performance Table (CAPM vs. FF3)")

            summary_data = []
            for stock in st.session_state.tickers:
                capm_r = st.session_state.capm_results.get(stock)
                ff3_r = st.session_state.ff3_results.get(stock)
                if capm_r and ff3_r:
                    summary_data.append({
                        "Stock": stock,
                        "CAPM_Alpha_Ann": f"{capm_r['alpha_ann']:.2%}",
                        "CAPM_Alpha_pvalue": f"{capm_r['alpha_pval']:.3f}",
                        "CAPM_Beta_M": f"{capm_r['beta_M']:.3f}",
                        "CAPM_R_squared": f"{capm_r['r_squared']:.3f}",
                        "CAPM_IR": f"{capm_r['information_ratio']:.3f}",
                        "FF3_Alpha_Ann": f"{ff3_r['alpha_ann']:.2%}",
                        "FF3_Alpha_pvalue": f"{ff3_r['alpha_pval']:.3f}",
                        "FF3_Beta_M": f"{ff3_r['beta_M']:.3f}",
                        "FF3_Beta_S": f"{ff3_r['beta_S']:.3f}",
                        "FF3_Beta_H": f"{ff3_r['beta_H']:.3f}",
                        "FF3_R_squared": f"{ff3_r['r_squared']:.3f}",
                        "FF3_Adj_R_squared": f"{ff3_r['adj_r_squared']:.3f}",
                        "FF3_IR": f"{ff3_r['information_ratio']:.3f}",
                        "R_squared_Improvement": f"{(ff3_r['r_squared'] - capm_r['r_squared']):.3f}",
                    })

            df_summary = pd.DataFrame(summary_data).set_index("Stock")
            st.dataframe(df_summary)

            st.markdown(
                "Interpretation prompt: treat each stock’s betas as a **factor fingerprint**. "
                "Compare fingerprints across holdings to understand concentration and unintended style tilts."
            )

            st.markdown("### Factor Beta Comparison Bar Chart")
            # Only include stocks that have FF3 results to ensure all arrays have the same length
            available_stocks = [
                s for s in st.session_state.tickers if s in st.session_state.ff3_results]

            if available_stocks:
                betas_df = pd.DataFrame({
                    "Stock": available_stocks,
                    "Beta_M": [st.session_state.ff3_results[s]["beta_M"] for s in available_stocks],
                    "Beta_S": [st.session_state.ff3_results[s]["beta_S"] for s in available_stocks],
                    "Beta_H": [st.session_state.ff3_results[s]["beta_H"] for s in available_stocks],
                })

                betas_melted = betas_df.melt(
                    id_vars="Stock", var_name="Factor", value_name="Beta")
                fig_beta_comp, ax_beta_comp = plt.subplots(figsize=(14, 7))
                sns.barplot(x="Stock", y="Beta", hue="Factor",
                            data=betas_melted, ax=ax_beta_comp)
                ax_beta_comp.set_title(
                    "Fama-French 3-Factor Betas Comparison Across Stocks")
                ax_beta_comp.set_xlabel("Stock")
                ax_beta_comp.set_ylabel("Factor Beta")
                ax_beta_comp.axhline(0, color="gray", linestyle="--")
                ax_beta_comp.legend(title="Factor", bbox_to_anchor=(
                    1.05, 1), loc="upper left")
                st.pyplot(fig_beta_comp)
                plt.close(fig_beta_comp)

                _decision_translation([
                    "If many holdings have similarly high βM, you have market-risk concentration even if names differ.",
                    "If βHML is broadly negative across holdings, you are implicitly running a growth book (watch value-rotation scenarios).",
                ])

            st.markdown("### Security Market Line (SML) Plot")
            avg_excess_returns = st.session_state.df_merged[[
                f"{s}_excess" for s in st.session_state.tickers]].mean()

            sml_data = []
            for stock in st.session_state.tickers:
                if stock in st.session_state.ff3_results:
                    sml_data.append({
                        "Stock": stock,
                        "Avg_Excess_Return": avg_excess_returns[f"{stock}_excess"],
                        "Market_Beta": st.session_state.ff3_results[stock]["beta_M"],
                    })

            df_sml = pd.DataFrame(sml_data)
            avg_mkt_rf = st.session_state.df_merged["Mkt_RF"].mean()
            theoretical_sml_x = np.linspace(
                df_sml["Market_Beta"].min() * 0.8, df_sml["Market_Beta"].max() * 1.2, 100)
            theoretical_sml_y = theoretical_sml_x * avg_mkt_rf * 12

            fig_sml, ax_sml = plt.subplots(figsize=(12, 7))
            df_sml_plot = df_sml.copy()
            df_sml_plot["Avg_Excess_Return_Ann"] = df_sml_plot["Avg_Excess_Return"] * 12

            sns.scatterplot(
                x="Market_Beta",
                y="Avg_Excess_Return_Ann",
                hue="Stock",
                data=df_sml_plot,
                s=100,
                zorder=2,
                ax=ax_sml,
            )

            ax_sml.plot(
                theoretical_sml_x,
                theoretical_sml_y,
                color="red",
                linestyle="--",
                label=f"Theoretical SML (E[Mkt-RF] Ann: {avg_mkt_rf*12:.2%})",
            )
            ax_sml.set_title(
                "Security Market Line (SML): Annualized Excess Returns vs. Market Beta")
            ax_sml.set_xlabel("Market Beta")
            ax_sml.set_ylabel("Annualized Average Excess Return")
            ax_sml.axhline(0, color="gray", linestyle="--", alpha=0.7)
            ax_sml.axvline(1, color="gray", linestyle=":",
                           alpha=0.7, label="Market Beta = 1")
            ax_sml.legend()
            st.pyplot(fig_sml)
            plt.close(fig_sml)

            _assumptions_box(
                "SML watch-outs (click to expand)",
                [
                    "This is an *in-sample* plot using realized average returns; it is not a forward guarantee.",
                    "Stocks above/below the line can reflect noise, regime-specific premia, or omitted risks.",
                ],
            )

        else:
            st.info(
                "Run CAPM and FF3 (at least for one stock each) to unlock comparison tables and plots.")

elif st.session_state.current_page == "Model Diagnostics":
    st.title("Model Diagnostics: When Are p-values and t-stats Meaningful?")

    st.markdown("### 4. Validating Model Assumptions: Diagnostic Tests")
    st.markdown(
        "Before relying on the factor model for critical investment decisions, Alex must perform diagnostic tests "
        "to check if the underlying assumptions of Ordinary Least Squares (OLS) regression are met. "
        "Violations of these assumptions (e.g., autocorrelation, heteroskedasticity, multicollinearity) can lead to "
        "inefficient or biased parameter estimates and incorrect statistical inferences (e.g., t-statistics, p-values). "
        "This step ensures the robustness of his analysis."
    )
    st.markdown(f"")

    st.markdown("He will check for:")
    st.markdown(f"")

    st.markdown(
        "**Autocorrelation (Durbin-Watson statistic):** Checks if residuals are correlated over time.")
    st.markdown(r"""
$$
DW \approx 2(1 - \rho_1)
$$
""")
    st.markdown(
        r"where $\rho_1$ is the first-order autocorrelation of residuals. "
        r"A value close to 2 indicates no autocorrelation. For financial time series, "
        r"positive autocorrelation ($DW < 2$, especially $< 1.5$) can indicate momentum effects or missing factors."
    )
    st.markdown(f"")

    st.markdown(
        "**Heteroskedasticity (Breusch-Pagan test):** Checks if the variance of the residuals is constant across all levels of independent variables. "
        "Heteroskedasticity (Breusch-Pagan p-value $< 0.05$) leads to inefficient estimates and incorrect standard errors."
    )
    st.markdown(f"")

    st.markdown("**Multicollinearity (Variance Inflation Factor - VIF):** Checks if independent variables are highly correlated with each other.")
    st.markdown(r"""
$$
VIF_j = \frac{1}{1 - R_j^2}
$$
""")
    st.markdown(
        r"where $R_j^2$ is the R-squared from regressing factor $j$ on all other factors. "
        r"High multicollinearity ($VIF > 5$ or $10$) can make coefficient estimates unstable and difficult to interpret."
    )
    st.markdown(f"")

    st.markdown(
        "**Practitioner Warning:** Heteroskedasticity is common in financial data. If the Breusch-Pagan test rejects homoskedasticity "
        "(typical for equity returns, where volatility clusters in crises), switch to Newey-West HAC standard errors for robust inference. "
        "HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors produce valid t-statistics and p-values even when classical OLS "
        "assumptions are violated—a critical technique for financial time-series."
    )
    st.markdown(f"")

    if not st.session_state.ff3_results:
        st.warning(
            "Please run Fama-French 3-Factor regressions on the previous page first.")
    else:
        selected_stock = st.selectbox(
            "Stock to diagnose (choose one and interpret deeply)",
            st.session_state.tickers,
            key="diag_stock_select",
            help="Diagnostics are most valuable when you connect them to what you trust (or don’t trust) in your inference."
        )

        if st.button(f"Run Diagnostics for {selected_stock}"):
            ff3_model = st.session_state.ff3_results[selected_stock]["model"]
            X_ff3_for_vif = st.session_state.df_merged[[
                "Mkt_RF", "SMB", "HML"]]

            with st.spinner(f"Running diagnostic tests for {selected_stock}'s FF3 model..."):
                try:
                    results = run_diagnostic_tests(ff3_model, X_ff3_for_vif)
                    st.session_state.diagnostic_results[selected_stock] = results
                    st.success("Diagnostics completed!")
                except Exception as e:
                    st.error(
                        f"Error running diagnostics for {selected_stock}: {e}")

        # Show the most recent results for the selected stock (if available)
        if (
            "diag_stock_select" in st.session_state
            and st.session_state.diagnostic_results.get(st.session_state.diag_stock_select)
        ):
            selected_stock = st.session_state.diag_stock_select
            results = st.session_state.diagnostic_results[selected_stock]

            st.subheader(f"Diagnostic Test Results for {selected_stock}")

            d1, d2, d3 = st.columns(3)
            d1.metric("Durbin–Watson", _fmt_num(results["dw_stat"]))
            d2.metric("Breusch–Pagan p-value",
                      _fmt_num(results["bp_pvalue"], decimals=4))
            d3.metric("Max VIF (across factors)", _fmt_num(
                max(results["vif_results"].values())))

            st.markdown(
                f"- **DW interpretation:** {results['dw_interpretation']}")
            st.markdown(
                f"- **BP interpretation:** {results['bp_interpretation']}")
            vif_str = ", ".join(
                [f"{k}: {v:.2f}" for k, v in results["vif_results"].items()])
            st.markdown(f"- **VIFs:** {vif_str}")
            st.markdown(
                f"- **VIF interpretation:** {results['vif_interpretation']}")

            _decision_translation([
                "If BP rejects homoskedasticity, treat classical OLS p-values with caution; prefer HAC-robust inference in real reporting.",
                "If DW suggests autocorrelation, your factor set may be missing a relevant driver (or you’re in a regime with persistence).",
                "If VIF is high, interpret individual factor betas carefully because they may be statistically unstable.",
            ])

            st.markdown("### Diagnostic 4-Panel Plot of Residuals")
            ff3_model = st.session_state.ff3_results[selected_stock]["model"]

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(
                f"Regression Diagnostic Plots for {selected_stock} (Fama-French 3-Factor Model)",
                fontsize=16
            )

            axes[0, 0].plot(ff3_model.resid.index, ff3_model.resid)
            axes[0, 0].axhline(y=0, color="red", linestyle="--")
            axes[0, 0].set_title("Residuals Over Time")
            axes[0, 0].set_ylabel("Residual Value")

            axes[0, 1].scatter(ff3_model.fittedvalues,
                               ff3_model.resid, alpha=0.5)
            axes[0, 1].axhline(y=0, color="red", linestyle="--")
            axes[0, 1].set_title("Residuals vs Fitted Values")
            axes[0, 1].set_xlabel("Fitted Values")
            axes[0, 1].set_ylabel("Residual Value")

            sm.qqplot(ff3_model.resid, line="45", ax=axes[1, 0])
            axes[1, 0].set_title("Q–Q Plot of Residuals")

            axes[1, 1].hist(ff3_model.resid, bins=30,
                            edgecolor="black", alpha=0.7)
            axes[1, 1].set_title("Residual Distribution")
            axes[1, 1].set_xlabel("Residual Value")
            axes[1, 1].set_ylabel("Frequency")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig)
            plt.close(fig)

elif st.session_state.current_page == "Rolling Betas":
    st.title("Rolling Betas: Detecting Regime Change in Factor Exposures")

    st.markdown("### 5. Dynamic Factor Exposures: Rolling Betas")
    st.markdown(
        "Static, full-sample betas can mask how a stock's sensitivity to factors changes over time, especially during different market regimes "
        "or significant economic events. As a Portfolio Manager, Alex needs to understand this dynamic nature for effective risk management and "
        "tactical asset allocation. He will compute and visualize rolling betas over a defined window (e.g., 36 months) to observe how these exposures evolve."
    )
    st.markdown("")
    st.markdown(
        "This technique involves running the factor regression repeatedly on a moving window of historical data. "
        "The resulting time series of betas provides insights into how the stock's \"factor fingerprint\" adapts to changing market conditions."
    )
    st.markdown("")

    _assumptions_box(
        "Rolling beta watch-outs (click to expand)",
        [
            "Short windows react faster but are noisier; long windows are smoother but can miss fast regime breaks.",
            "Rolling estimates can be distorted by single extreme observations in short windows (crisis months).",
        ],
    )

    if not _has_data_ready():
        st.warning(
            "Please retrieve data on the 'Introduction & Data Setup' page first.")
    else:
        selected_stock_rb = st.selectbox(
            "Stock to analyze (rolling betas)",
            st.session_state.tickers,
            key="rolling_beta_stock_select",
            help="Pick a stock where you suspect beta changes across regimes (e.g., TSLA around 2020–2022)."
        )

        st.session_state.rolling_window_size = st.slider(
            "Rolling window length (months)",
            min_value=12,
            max_value=60,
            value=st.session_state.rolling_window_size,
            step=6,
            help="36 months is a common compromise for monthly data."
        )

        if st.button(f"Calculate & Plot Rolling Betas for {selected_stock_rb}"):
            with st.spinner(f"Calculating rolling {st.session_state.rolling_window_size}-month betas for {selected_stock_rb}..."):
                try:
                    rolling_betas_df = calculate_rolling_betas(
                        st.session_state.df_merged,
                        selected_stock_rb,
                        st.session_state.rolling_window_size
                    )

                    st.session_state.last_rolling_betas_stock = selected_stock_rb

                    st.markdown("### Rolling Beta Time-Series Plot")
                    fig, ax = plt.subplots(figsize=(14, 7))
                    rolling_betas_df.plot(ax=ax)

                    ax.set_title(
                        f"Rolling {st.session_state.rolling_window_size}-Month FF3 Betas for {selected_stock_rb}"
                    )
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Beta Value")
                    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
                    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)

                    # Annotate significant market events
                    plot_start = rolling_betas_df.index.min()
                    plot_end = rolling_betas_df.index.max()

                    if pd.Timestamp("2020-02-01") >= plot_start and pd.Timestamp("2020-04-01") <= plot_end:
                        ax.axvspan(
                            datetime.datetime(2020, 2, 1),
                            datetime.datetime(2020, 4, 1),
                            color="red",
                            alpha=0.2,
                            label="COVID-19 Crash"
                        )

                    if pd.Timestamp("2022-01-01") >= plot_start and pd.Timestamp("2022-12-01") <= plot_end:
                        ax.axvspan(
                            datetime.datetime(2022, 1, 1),
                            datetime.datetime(2022, 12, 1),
                            color="purple",
                            alpha=0.1,
                            label="Inflation/Rate Hikes"
                        )

                    ax.legend(title="Factor", bbox_to_anchor=(
                        1.05, 1), loc="upper left")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.success("Rolling beta analysis completed!")

                    _decision_translation([
                        "If βM rises in stress windows, your portfolio drawdown may be worse than implied by full-sample beta.",
                        "If βHML flips sign across time, your style exposure is regime-dependent—stress test value/growth rotations.",
                    ])

                except Exception as e:
                    st.error(
                        f"Error calculating or plotting rolling betas for {selected_stock_rb}: {e}")

        if selected_stock_rb and st.session_state.df_merged is not None:
            st.markdown(
                f"Interpretation cue: focus on whether {selected_stock_rb}'s betas are stable or regime-dependent. "
                f"Large swings are often more decision-relevant than the full-sample point estimate."
            )

elif st.session_state.current_page == "Scenario Analysis":
    st.title("Scenario Analysis: Translating Betas into Forward-Looking Stress Tests")

    st.markdown("### 6. Forward-Looking Analysis: Scenario Projections")
    st.markdown(
        "One of the most powerful applications of factor models for Alex is to project expected returns under various hypothetical macroeconomic scenarios. "
        "This shifts the analysis from purely backward-looking performance attribution to a forward-looking risk management and strategic planning tool."
    )
    st.markdown("")
    st.markdown(
        "By defining reasonable expected returns for the Fama-French factors in different economic environments, "
        "Alex can estimate how his target stocks might perform."
    )
    st.markdown(
        "The scenario projection uses the estimated betas from the Fama-French 3-factor model:")
    st.markdown(
        r"""
$$
E[R_i - R_f] = \hat{\alpha}_i + \hat{\beta}_{i,M} E[R_m - R_f] + \hat{\beta}_{i,S} E[SMB] + \hat{\beta}_{i,H} E[HML]
$$
"""
    )
    st.markdown(
        r"where $E[...]$ denotes the expected value of the factors under a specific scenario, and $\hat{\alpha}$, $\hat{\beta}$ are the estimated coefficients from the full-sample regression."
    )
    st.markdown("Alex will define several plausible scenarios and then calculate the projected annualized excess return for each stock.")
    st.markdown("")

    _assumptions_box(
        "Scenario guardrails (click to expand)",
        [
            "Scenarios are **inputs**, not outputs. The tool helps you reason consistently, not predict with certainty.",
            "Factor expectations should be economically coherent (e.g., crash regimes often coincide with negative Mkt-RF).",
            "Projections are conditional on model stability and the assumption that betas remain relevant in the scenario horizon.",
        ],
    )

    if not st.session_state.ff3_results:
        st.warning(
            "Please run Fama-French 3-Factor regressions on the FF3 page first.")
    else:
        st.subheader(
            "Define Macroeconomic Scenarios (Monthly Expected Factor Returns)")

        st.caption(
            "Tip: inputs are **monthly** expected factor returns. The output is reported as **annualized excess return**."
        )

        current_scenarios_copy = st.session_state.macro_scenarios.copy()
        scenarios_to_delete = []

        for scenario_name in list(current_scenarios_copy.keys()):
            st.markdown(f"**Scenario: {scenario_name}**")
            col_mkt, col_smb, col_hml, col_del = st.columns(
                [0.3, 0.3, 0.3, 0.1])

            with col_mkt:
                current_scenarios_copy[scenario_name]["Mkt_RF"] = st.number_input(
                    "Mkt-RF (monthly)",
                    value=current_scenarios_copy[scenario_name]["Mkt_RF"],
                    format="%.4f",
                    key=f"{scenario_name}_Mkt_RF",
                    help="Example: 0.0060 ≈ 0.6% monthly market excess return."
                )
            with col_smb:
                current_scenarios_copy[scenario_name]["SMB"] = st.number_input(
                    "SMB (monthly)",
                    value=current_scenarios_copy[scenario_name]["SMB"],
                    format="%.4f",
                    key=f"{scenario_name}_SMB",
                    help="Small-cap premium proxy (monthly)."
                )
            with col_hml:
                current_scenarios_copy[scenario_name]["HML"] = st.number_input(
                    "HML (monthly)",
                    value=current_scenarios_copy[scenario_name]["HML"],
                    format="%.4f",
                    key=f"{scenario_name}_HML",
                    help="Value premium proxy (monthly)."
                )
            with col_del:
                if st.button("Delete", key=f"delete_{scenario_name}"):
                    scenarios_to_delete.append(scenario_name)

            # Show annualized equivalents as immediate intuition
            ann_mkt = (
                1 + current_scenarios_copy[scenario_name]["Mkt_RF"]) ** 12 - 1
            ann_smb = (
                1 + current_scenarios_copy[scenario_name]["SMB"]) ** 12 - 1
            ann_hml = (
                1 + current_scenarios_copy[scenario_name]["HML"]) ** 12 - 1
            st.caption(
                f"Annualized intuition (approx): Mkt-RF {_fmt_pct(ann_mkt)}, SMB {_fmt_pct(ann_smb)}, HML {_fmt_pct(ann_hml)}"
            )
            st.markdown("---")

        if scenarios_to_delete:
            for sc_name in scenarios_to_delete:
                if sc_name in st.session_state.macro_scenarios:
                    del st.session_state.macro_scenarios[sc_name]
            st.success("Scenario(s) deleted. Please re-run projection.")
            st.rerun()

        st.session_state.macro_scenarios = current_scenarios_copy

        with st.expander("Add New Scenario"):
            new_scenario_name = st.text_input(
                "New Scenario Name:", key="new_scenario_name")
            new_mkt = st.number_input(
                "New Mkt-RF (monthly):", value=0.005, format="%.4f", key="new_mkt")
            new_smb = st.number_input(
                "New SMB (monthly):", value=0.001, format="%.4f", key="new_smb")
            new_hml = st.number_input(
                "New HML (monthly):", value=0.000, format="%.4f", key="new_hml")
            if st.button("Add Scenario") and new_scenario_name:
                if new_scenario_name not in st.session_state.macro_scenarios:
                    st.session_state.macro_scenarios[new_scenario_name] = {
                        "Mkt_RF": new_mkt, "SMB": new_smb, "HML": new_hml}
                    st.success(f"Scenario '{new_scenario_name}' added!")
                    st.rerun()
                else:
                    st.warning(
                        f"Scenario '{new_scenario_name}' already exists. Please choose a different name.")

        if st.button("Project Returns Under Scenarios"):
            st.session_state.scenario_projections = {}

            for stock in st.session_state.tickers:
                if stock not in st.session_state.ff3_results:
                    continue
                ff3_model_params = st.session_state.ff3_results[stock]["model"].params
                projections = project_returns_under_scenarios(
                    ff3_model_params, st.session_state.macro_scenarios)
                st.session_state.scenario_projections[stock] = projections

            st.success("Scenario projections completed!")

            st.markdown(
                "### Summary of Projected Annualized Excess Returns Across All Stocks")
            all_projections_df = pd.DataFrame()

            for stock, projections in st.session_state.scenario_projections.items():
                projections_indexed = projections.set_index("Scenario").rename(
                    columns={"Projected_Annual_Excess_Return": stock})
                if all_projections_df.empty:
                    all_projections_df = projections_indexed
                else:
                    all_projections_df = all_projections_df.join(
                        projections_indexed)

            st.dataframe(all_projections_df.applymap(lambda x: f"{x:.2%}"))

            _decision_translation([
                "If a stock shows large downside in ‘Market Crash’, it is a candidate for hedging, sizing down, or pairing with defensive exposures.",
                "If a stock benefits disproportionately from ‘Value Rotation’, it may serve as a portfolio ballast when growth leadership reverses.",
            ])

    if st.session_state.scenario_projections:
        st.markdown(
            "Interpretation cue: compare *relative* projections across names to understand factor-driven vulnerability and diversification. "
            "Absolute magnitudes depend heavily on scenario inputs."
        )

elif st.session_state.current_page == "Performance Attribution & Report":
    st.title("Performance Attribution & Report")

    st.markdown("### 7. Performance Attribution and Model Summary")
    st.markdown(
        "To complete his comprehensive analysis, Alex wants to visualize the contribution of each factor to the cumulative excess return of a stock. "
        "This \"cumulative return decomposition\" helps him attribute performance to market, size, and value factors versus the stock's idiosyncratic alpha. "
        "He also wants a visual comparison of the model's predicted versus actual returns and a final summary of all key metrics for easy reporting."
    )
    st.markdown("")

    st.markdown(
        "The cumulative contribution of each factor at time $T$ is given by:")
    st.markdown(
        r"""
$$
\text{Cumulative Factor Contribution}_X = \sum_{t=1}^T \hat{\beta}_{X} \cdot F_{X,t}
$$
""")
    st.markdown(
        r"where $F_{X,t}$ is the factor return for factor $X$ at time $t$. The cumulative alpha contribution is $\sum_{t=1}^T \hat{\alpha}$."
    )
    st.markdown("")

    if not (st.session_state.ff3_results and _has_data_ready()):
        st.warning(
            "Please run Fama-French 3-Factor regressions on the FF3 page first.")
    else:
        selected_stock_pa = st.selectbox(
            "Select stock for performance attribution:",
            st.session_state.tickers,
            key="perf_attr_stock_select"
        )

        if selected_stock_pa not in st.session_state.ff3_results:
            st.info("Run FF3 for this stock to enable attribution.")
        else:
            if st.button(f"Generate Performance Attribution for {selected_stock_pa}"):
                ff3_model = st.session_state.ff3_results[selected_stock_pa]["model"]
                ff3_model_params = ff3_model.params

                st.markdown("### Cumulative Return Decomposition Chart")
                y = st.session_state.df_merged[f"{selected_stock_pa}_excess"]
                X = st.session_state.df_merged[["Mkt_RF", "SMB", "HML"]]

                alpha = ff3_model_params["const"]
                beta_M = ff3_model_params["Mkt_RF"]
                beta_S = ff3_model_params["SMB"]
                beta_H = ff3_model_params["HML"]

                market_contribution = beta_M * X["Mkt_RF"]
                smb_contribution = beta_S * X["SMB"]
                hml_contribution = beta_H * X["HML"]
                alpha_contribution = pd.Series(
                    alpha, index=st.session_state.df_merged.index)

                total_model_return = market_contribution + \
                    smb_contribution + hml_contribution + alpha_contribution
                epsilon_contribution = y - total_model_return

                attribution_df = pd.DataFrame({
                    "Market Factor": market_contribution.cumsum(),
                    "SMB Factor": smb_contribution.cumsum(),
                    "HML Factor": hml_contribution.cumsum(),
                    "Alpha": alpha_contribution.cumsum(),
                    "Residual (Unexplained)": epsilon_contribution.cumsum(),
                    "Actual Excess Return": y.cumsum(),
                }, index=st.session_state.df_merged.index)

                fig_cum_decomp, ax_cum_decomp = plt.subplots(figsize=(14, 7))
                attribution_df[["Market Factor", "SMB Factor", "HML Factor", "Alpha", "Residual (Unexplained)"]].plot(
                    kind="area", stacked=False, ax=ax_cum_decomp, alpha=0.7
                )
                attribution_df["Actual Excess Return"].plot(
                    ax=ax_cum_decomp, color="black", linestyle="--", linewidth=2, label="Actual Excess Return"
                )

                ax_cum_decomp.set_title(
                    f"Cumulative Return Decomposition for {selected_stock_pa} (FF3)")
                ax_cum_decomp.set_xlabel("Date")
                ax_cum_decomp.set_ylabel("Cumulative Return")
                ax_cum_decomp.legend(
                    title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                st.pyplot(fig_cum_decomp)
                plt.close(fig_cum_decomp)

                _decision_translation([
                    "If ‘Residual (Unexplained)’ dominates, your factor model is not capturing key drivers—avoid overconfident attribution narratives.",
                    "If ‘Market Factor’ dominates, the position behaves largely as market exposure (think sizing, hedging, or substitutability).",
                    "If ‘Alpha’ is small but ‘Actual’ is large, performance likely came from factor timing/regime rather than persistent skill.",
                ])

                st.markdown("### Predicted vs. Actual Scatter Plot")
                y_actual = st.session_state.df_merged[f"{selected_stock_pa}_excess"]
                X_pred = sm.add_constant(
                    st.session_state.df_merged[["Mkt_RF", "SMB", "HML"]])
                y_predicted = ff3_model.predict(X_pred)

                fig_pred_actual, ax_pred_actual = plt.subplots(figsize=(8, 8))
                sns.scatterplot(x=y_predicted, y=y_actual,
                                alpha=0.6, ax=ax_pred_actual)

                min_val = min(y_actual.min(), y_predicted.min())
                max_val = max(y_actual.max(), y_predicted.max())

                ax_pred_actual.plot(
                    [min_val, max_val], [min_val, max_val],
                    color="red", linestyle="--",
                    label="45-degree line (Perfect Prediction)"
                )
                ax_pred_actual.set_title(
                    f"Predicted vs. Actual Excess Returns for {selected_stock_pa}")
                ax_pred_actual.set_xlabel("Predicted Excess Return")
                ax_pred_actual.set_ylabel("Actual Excess Return")
                ax_pred_actual.legend()
                ax_pred_actual.grid(True)
                plt.tight_layout()
                st.pyplot(fig_pred_actual)
                plt.close(fig_pred_actual)

                st.success("Performance attribution plots generated!")

    # Final report table (kept) but framed to prevent misuse
    if st.session_state.ff3_results and st.session_state.capm_results and st.session_state.diagnostic_results and st.session_state.tickers:
        st.markdown("### Comprehensive Factor Exposure & Performance Report")

        final_summary_data = []
        for stock in st.session_state.tickers:
            capm_r = st.session_state.capm_results.get(stock)
            ff3_r = st.session_state.ff3_results.get(stock)
            diag_r = st.session_state.diagnostic_results.get(stock)

            if capm_r and ff3_r and diag_r:
                r_squared_improvement = ff3_r["r_squared"] - \
                    capm_r["r_squared"]
                vifs_str = ", ".join(
                    [f"{k}: {v:.2f}" for k, v in diag_r["vif_results"].items()])

                final_summary_data.append({
                    "Stock": stock,
                    "FF3_Alpha_Ann (%)": ff3_r["alpha_ann"] * 100,
                    "FF3_Alpha_pvalue": ff3_r["alpha_pval"],
                    "FF3_Beta_M": ff3_r["beta_M"],
                    "FF3_Beta_S": ff3_r["beta_S"],
                    "FF3_Beta_H": ff3_r["beta_H"],
                    "FF3_R_squared": ff3_r["r_squared"],
                    "R2_Improvement (FF3-CAPM)": r_squared_improvement,
                    "FF3_IR": ff3_r["information_ratio"],
                    "DW_Stat": diag_r["dw_stat"],
                    "BP_Pvalue": diag_r["bp_pvalue"],
                    "VIFs": vifs_str,
                })

        df_final_report = pd.DataFrame(final_summary_data).set_index("Stock")

        # Format for readability
        df_final_report_formatted = df_final_report.copy()
        for col in ["FF3_Alpha_Ann (%)"]:
            if col in df_final_report_formatted.columns:
                df_final_report_formatted[col] = df_final_report_formatted[col].map(
                    "{:.2f}%".format)

        for col in [
            "FF3_Alpha_pvalue", "FF3_Beta_M", "FF3_Beta_S", "FF3_Beta_H",
            "FF3_R_squared", "R2_Improvement (FF3-CAPM)", "FF3_IR", "DW_Stat"
        ]:
            if col in df_final_report_formatted.columns:
                df_final_report_formatted[col] = df_final_report_formatted[col].map(
                    "{:.3f}".format)

        if "BP_Pvalue" in df_final_report_formatted.columns:
            df_final_report_formatted["BP_Pvalue"] = df_final_report_formatted["BP_Pvalue"].map(
                "{:.4f}".format)

        st.dataframe(df_final_report_formatted)

        st.warning(
            "Guardrail: do not treat this table as a ‘ranking’ output. "
            "Use it to (i) diagnose risk concentration, (ii) test robustness, and (iii) communicate assumptions."
        )

        st.markdown(
            "This workflow gives a reproducible, interpretable baseline for beta estimation and factor exposure analysis—"
            "useful for risk committees, portfolio construction discussions, and stress-testing narratives."
        )

# License (do not modify)
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
