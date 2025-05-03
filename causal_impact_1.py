import requests
import streamlit as st
import pandas as pd
from causalimpact import CausalImpact
import matplotlib.pyplot as plt
import warnings
from io import BytesIO

# Page config
st.set_page_config(page_title="MMM and Causal Impact working together", layout="wide")

#MARKETING MIX MODELING
st.markdown("# 📊 Google Meridian Estimations", unsafe_allow_html=True)
# Block of text right after the HTML content
st.markdown("""
### 📝 **Analysis Overview**

This section presents the analysis from the **Google Meridian** tool. It includes key insights and recommendations derived from the data from the priors ROI.

Paid Search (2.6), Paid Social (4.4), OTA agreements (3.0), Paid Display (3.3).

The analysis demonstrates how these channels contribute to the overall revenue across different markets.

Feel free to explore the full data and results!
""")
# Fetch and embed the HTML content from GitHub
html_url = "https://raw.githubusercontent.com/kaalba/mmm-ci-op25/main/summary_output_v2.html"
response = requests.get(html_url)
html_content = response.text

# Embed in Streamlit
st.components.v1.html(html_content, height=600, scrolling=True)

#CAUSAL IMPACT - BRAND PAUSE

st.markdown("# 📉 Causal Impact Analysis (Brand Pause)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dummy_data_ci.csv", parse_dates=["date"])
    return df

df = load_data()

# Validate columns
required_columns = {"date", "market", "conversions", "Paid Search", "Paid Social"}
if not required_columns.issubset(df.columns):
    st.error(f"Missing columns in CSV. Required: {required_columns}")
    st.stop()

# Input: pause date
pause_date = '2024-07-28'

# Input: market selection
markets = df["market"].unique()
selected_market = st.selectbox("🌍 Select Market", markets, key ="market_select_1")

# Filter and prepare data
df_m = df[df["market"] == selected_market].copy()
df_m = df_m[["date", "Paid Search", "Paid Social", "conversions"]]
df_m = df_m.set_index("date")

full_weeks = df_m.index.sort_values().unique()
pre_period = [full_weeks.min(), full_weeks[full_weeks < pd.to_datetime(pause_date)].max()]
post_period = [pd.to_datetime(pause_date), full_weeks.max()]

ci_data = df_m[["conversions", "Paid Search", "Paid Social"]]
impact = CausalImpact(ci_data, pre_period, post_period)

# Output
# Extract components
summary = impact.summary_data.round(2)

# Pull values using correct labels
avg_effect = summary.at["abs_effect", "average"]
rel_effect = summary.at["rel_effect", "average"]
ci_lower = summary.at["rel_effect_lower", "average"]
ci_upper = summary.at["rel_effect_upper", "average"]

# Format for display
rel_effect_fmt = f"{rel_effect:.2%}"
ci_fmt = f"{ci_lower:.2%} to {ci_upper:.2%}"
avg_effect_fmt = f"{avg_effect:,.2f}"

# Display with metrics
st.subheader("🧮 **Key Impact Metrics**")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📉 Avg Effect", avg_effect_fmt)

with col2:
    st.metric("📊 Relative Effect", rel_effect_fmt)

with col3:
    st.metric("🎯 95% CI", ci_fmt)

with st.expander("📋 Show Full Numerical Results"):
    st.dataframe(summary.round(2))

with st.expander("📝 Full Explanation Report"):
    st.markdown(f"```{impact.summary(output='report')}```")

st.subheader("📈 Impact Plot - All Plots")
# Get date index and inference data
inferences = impact.inferences.copy()
inferences["date"] = pd.date_range(start="2024-01-01", periods=52, freq='W')
inferences.set_index("date", inplace=True)

inferences["actual"] = impact.data["conversions"]
inferences = inferences.iloc[1:].fillna(0)

# Y-axis ranges
ymin_1 = inferences["actual"].min() - 100
ymax_1 = inferences["actual"].max() + 100
ymin_2 = inferences["point_effects"].min() - 100
ymax_2 = inferences["point_effects"].max() + 100
ymin_3 = inferences["post_cum_effects"].min() - 100
ymax_3 = inferences["post_cum_effects"].max() + 100

# 📈 Plot 1: Actual vs Predicted
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(inferences.index, inferences["actual"], label="Actual", color="black")
ax1.plot(inferences.index, inferences["preds"], label="Predicted", linestyle="--", color="blue")
ax1.axvline(pd.to_datetime(pause_date), linestyle="--", color="gray")
ax1.fill_between(inferences.index, inferences["preds_lower"], inferences["preds_upper"], color="blue", alpha=0.2)
ax1.set_ylim(ymin_1, ymax_1)
ax1.set_title(f"Actual vs Predicted")
ax1.legend()
st.pyplot(fig1)

# 📊 Plot 2: Pointwise Effects
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(inferences.index, inferences["point_effects"], label="Pointwise Effect", color="purple")
ax2.axhline(0, linestyle="--", color="gray")
ax2.axvline(pd.to_datetime(pause_date), linestyle="--", color="gray")
ax2.fill_between(inferences.index, inferences["point_effects_lower"], inferences["point_effects_upper"], color="purple", alpha=0.2)
ax2.set_ylim(ymin_2, ymax_2)
ax2.set_title(f"Pointwise Effects")
ax2.legend()
st.pyplot(fig2)

# 📉 Plot 3: Cumulative Effects
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(inferences.index, inferences["post_cum_effects"], label="Cumulative Effect", color="green")
ax3.axhline(0, linestyle="--", color="gray")
ax3.axvline(pd.to_datetime(pause_date), linestyle="--", color="gray")
ax3.fill_between(inferences.index, inferences["post_cum_effects_lower"], inferences["post_cum_effects_upper"], color="green", alpha=0.2)
ax3.set_ylim(ymin_3, ymax_3)
ax3.set_title(f"Cumulative Effects")
ax3.legend()
st.pyplot(fig3)

#MARKETING MIX MODELING - BUDGET OPTIMIZATION
st.markdown("# 📊 Google Meridian Budget Optimization", unsafe_allow_html=True)
# Block of text right after the HTML content
st.markdown("""
### 📝 **Analysis Overview**

This section presents the analysis from the **Google Meridian** budget optimizer tool.

It includes key insights and recommendations after the results from the causal impact test.

The analysis demonstrates how shifting the budget to **Paid Display** Improves the overall revenue.

Feel free to explore the full data and results!
""")
# Fetch and embed the HTML content from GitHub
html_url = "https://raw.githubusercontent.com/kaalba/mmm-ci-op25/main/optimization_output.html"
response = requests.get(html_url)
html_content = response.text

# Embed in Streamlit
st.components.v1.html(html_content, height=600, scrolling=True)

#CAUSAL IMPACT - BUDGET SHIFT

st.markdown("# 📉 Causal Impact Analysis (Budget Optimization)")

# Load data
@st.cache_data
def load_data():
    df2 = pd.read_csv("dummy_data_ci2.csv", parse_dates=["date"])
    return df2

df2 = load_data()

# Validate columns
required_columns = {"date", "market", "conversions", "Paid Search", "Paid Social"}
if not required_columns.issubset(df.columns):
    st.error(f"Missing columns in CSV. Required: {required_columns}")
    st.stop()

# Input: pause date
pause_date = pd.to_datetime('2025-02-09')

# Input: market selection
markets = df["market"].unique()
selected_market = st.selectbox("🌍 Select Market", markets, key ="market_select_2")

# Filter and prepare data
df_m = df2[df2["market"] == selected_market].copy()
df_m = df_m[["date", "Paid Search", "Paid Social", "conversions"]]
df_m = df_m.set_index("date")

full_weeks = df_m.index.sort_values().unique()
pre_period = [full_weeks.min(), full_weeks[full_weeks < pd.to_datetime(pause_date)].max()]
post_period = [pd.to_datetime(pause_date), full_weeks.max()]

ci_data = df_m[["conversions", "Paid Search", "Paid Social"]]
impact = CausalImpact(ci_data, pre_period, post_period)

# Output
# Extract components
summary = impact.summary_data.round(2)

# Pull values using correct labels
avg_effect = summary.at["abs_effect", "average"]
rel_effect = summary.at["rel_effect", "average"]
ci_lower = summary.at["rel_effect_lower", "average"]
ci_upper = summary.at["rel_effect_upper", "average"]

# Format for display
rel_effect_fmt = f"{rel_effect:.2%}"
ci_fmt = f"{ci_lower:.2%} to {ci_upper:.2%}"
avg_effect_fmt = f"{avg_effect:,.2f}"

# Display with metrics
st.subheader("🧮 **Key Impact Metrics**")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📉 Avg Effect", avg_effect_fmt)

with col2:
    st.metric("📊 Relative Effect", rel_effect_fmt)

with col3:
    st.metric("🎯 95% CI", ci_fmt)

with st.expander("📋 Show Full Numerical Results"):
    st.dataframe(summary.round(2))

with st.expander("📝 Full Explanation Report"):
    st.markdown(f"```{impact.summary(output='report')}```")

st.subheader("📈 Impact Plot - All Plots")
# Get date index and inference data
inferences = impact.inferences.copy()
inferences["date"] = pd.date_range(start="2024-05-05", periods=52, freq='W')
inferences.set_index("date", inplace=True)

inferences["actual"] = impact.data["conversions"]
inferences = inferences.iloc[1:].fillna(0)

# Y-axis ranges
ymin_1 = inferences["actual"].min() - 100
ymax_1 = inferences["actual"].max() + 100
ymin_2 = inferences["point_effects"].min() - 100
ymax_2 = inferences["point_effects"].max() + 100
ymin_3 = inferences["post_cum_effects"].min() - 100
ymax_3 = inferences["post_cum_effects"].max() + 100

# 📈 Plot 1: Actual vs Predicted
fig4, ax4 = plt.subplots(figsize=(12, 4))
ax4.plot(inferences.index, inferences["actual"], label="Actual", color="black")
ax4.plot(inferences.index, inferences["preds"], label="Predicted", linestyle="--", color="blue")
ax4.axvline(pd.to_datetime(pause_date), linestyle="--", color="gray")
ax4.fill_between(inferences.index, inferences["preds_lower"], inferences["preds_upper"], color="blue", alpha=0.2)
ax4.set_ylim(ymin_1, ymax_1)
ax4.set_title(f"Actual vs Predicted")
ax4.legend()
st.pyplot(fig4)

# 📊 Plot 2: Pointwise Effects
fig5, ax5 = plt.subplots(figsize=(12, 4))
ax5.plot(inferences.index, inferences["point_effects"], label="Pointwise Effect", color="purple")
ax5.axhline(0, linestyle="--", color="gray")
ax5.axvline(pd.to_datetime(pause_date), linestyle="--", color="gray")
ax5.fill_between(inferences.index, inferences["point_effects_lower"], inferences["point_effects_upper"], color="purple", alpha=0.2)
ax5.set_ylim(ymin_2, ymax_2)
ax5.set_title(f"Pointwise Effects")
ax5.legend()
st.pyplot(fig5)

# 📉 Plot 3: Cumulative Effects
fig6, ax6 = plt.subplots(figsize=(12, 4))
ax6.plot(inferences.index, inferences["post_cum_effects"], label="Cumulative Effect", color="green")
ax6.axhline(0, linestyle="--", color="gray")
ax6.axvline(pd.to_datetime(pause_date), linestyle="--", color="gray")
ax6.fill_between(inferences.index, inferences["post_cum_effects_lower"], inferences["post_cum_effects_upper"], color="green", alpha=0.2)
ax6.set_ylim(ymin_3, ymax_3)
ax6.set_title(f"Cumulative Effects")
ax6.legend()
st.pyplot(fig6)
