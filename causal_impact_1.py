import requests
import streamlit as st
import pandas as pd
from causalimpact import CausalImpact
import matplotlib.pyplot as plt
from io import BytesIO

# Page config
st.set_page_config(page_title="MMM and Causal Impact working together", layout="wide")

#MARKETING MIX MODELING
#Embedding HTML file 
# URL of the raw HTML file on GitHub
html_url = "https://raw.githubusercontent.com/kaalba/mmm-ci-op25/main/summary_output_v2.html"

# Fetch the HTML content
response = requests.get(html_url)
html_content = response.text

# Embed in Streamlit
st.title("## 📊 Google Meridian Analysis", unsafe_allow_html=True)
st.components.v1.html(html_content, height=600, scrolling=True)

#CAUSAL IMPACT - BRAND PAUSE

st.title("📉 Causal Impact Analysis (Brand Pause)")

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
selected_market = st.selectbox("🌍 Select Market", markets)

# Filter and prepare data
df_m = df[df["market"] == selected_market].copy()
df_m = df_m[["date", "Paid Search", "Paid Social", "conversions"]]
df_m = df_m.set_index("date")

full_weeks = df_m.index.sort_values().unique()
pre_period = [full_weeks.min(), full_weeks[full_weeks < pd.to_datetime(pause_date)].max()]
post_period = [pd.to_datetime(pause_date), full_weeks.max()]

ci_data = df_m[["conversions", "Paid Search", "Paid Social"]]
impact = CausalImpact(ci_data, pre_period, post_period)
results = impact.inferences

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
st.markdown("### 🧮 **Key Impact Metrics**")
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
# Plot the results using the CausalImpact plot function
fig = impact.plot()  # This returns a matplotlib Figure object
# Embed the plot into Streamlit
st.pyplot(fig)
