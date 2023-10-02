import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import zscore

from utils.info_messages import get_key_error_message_info
from utils.stats import modified_zscore

st.sidebar.info(get_key_error_message_info())

st.title("Exercício 2 - Z-Score e Modified Z-Score em *Temperatura* e *Radiação*")
st.subheader("Visualização do Dataset")

if "weather_df" not in st.session_state:
    weather_df = pd.read_csv("datasets/mediadiaria.csv")
    st.session_state["weather_df"] = weather_df

weather_df = st.session_state.get("weather_df")
st.dataframe(weather_df)

st.subheader("Temperatura")
threshold_temperature = st.number_input(
    "Valor do Threshold", value=3.0, step=0.5, format="%.2f"
)

weather_df["zscore_temperature"] = zscore(weather_df["temp"], ddof=0)
weather_df["mod_zscore_temperature"] = modified_zscore(weather_df["temp"])[0]

zscore_data = (
    weather_df["zscore_temperature"].copy().sort_values(ascending=False).values
)
ranks = np.linspace(1, len(zscore_data), len(zscore_data))
mask_outlier = zscore_data < threshold_temperature

# Gráfico
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 8))

## Z-Score

# No anomaly data
ax0.plot(
    ranks[mask_outlier],
    zscore_data[mask_outlier],
    "o",
    color="b",
    label="Cool Temperature",
)

# Anomaly data
ax0.plot(
    ranks[~mask_outlier],
    zscore_data[~mask_outlier],
    "o",
    color="r",
    label="Anomalies",
)
ax0.axhline(threshold_temperature, color="r", label="threshold", alpha=0.5)

ax0.set_title("Z-Score vs. Temperature", fontweight="bold")
ax0.set_xlabel("Temperature")
ax0.set_ylabel("Z-Score")
ax0.legend(loc="best")

## Modified Z-Score
mod_zscore_data = (
    weather_df["mod_zscore_temperature"].copy().sort_values(ascending=False).values
)
ranks = np.linspace(1, len(mod_zscore_data), len(mod_zscore_data))
mask_outlier = mod_zscore_data < threshold_temperature

# No anomaly data
ax1.plot(
    ranks[mask_outlier],
    mod_zscore_data[mask_outlier],
    "o",
    color="b",
    label="Cool Temperature",
)

# Anomaly data
ax1.plot(
    ranks[~mask_outlier],
    mod_zscore_data[~mask_outlier],
    "o",
    color="r",
    label="Anomalies",
)
ax1.axhline(threshold_temperature, color="r", label="threshold", alpha=0.5)

ax1.set_title("Modified Z-Score vs. Temperature", fontweight="bold")
ax1.set_xlabel("Temperature")
ax1.set_ylabel("Modified Z-Score")
ax1.legend(loc="best")

st.pyplot(fig)

with st.expander(label="Anomalias detectadas", expanded=True):
    st.write("Z-Score")
    st.dataframe(
        weather_df[weather_df["zscore_temperature"] > threshold_temperature][
            ["ano", "mes", "dia", "temp", "zscore_temperature"]
        ]
    )

    st.write("Modified Z-Score")
    st.dataframe(
        weather_df[weather_df["mod_zscore_temperature"] > threshold_temperature][
            ["ano", "mes", "dia", "temp", "mod_zscore_temperature"]
        ]
    )

st.subheader("Radiação")
threshold_radiation = st.number_input(
    "Valor do Threshold", value=-4.0, step=0.5, format="%.2f"
)

weather_df["zscore_radiation"] = zscore(weather_df["radiacao"], ddof=0)
weather_df["mod_zscore_radiation"] = modified_zscore(weather_df["radiacao"])[0]

zscore_data = weather_df["zscore_radiation"].copy().sort_values(ascending=False).values
ranks = np.linspace(1, len(zscore_data), len(zscore_data))
mask_outlier = zscore_data < threshold_radiation

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 8))

## Z-Score
# No anomaly data
ax0.plot(
    ranks[~mask_outlier],
    zscore_data[~mask_outlier],
    "o",
    color="b",
    label="Cool Radiation",
)

# Anomaly data
ax0.plot(
    ranks[mask_outlier],
    zscore_data[mask_outlier],
    "o",
    color="r",
    label="Anomalies",
)

ax0.axhline(threshold_radiation, color="r", label="threshold", alpha=0.5)
ax0.set_title("Z-Score vs. Radiation", fontweight="bold")
ax0.set_xlabel("Radiation")
ax0.set_ylabel("Z-Score")

ax0.legend(loc="best")

## Modified Z-Score
mod_zscore_data = (
    weather_df["mod_zscore_radiation"].copy().sort_values(ascending=False).values
)
ranks = np.linspace(1, len(mod_zscore_data), len(mod_zscore_data))
mask_outlier = mod_zscore_data < threshold_radiation

# No anomaly data
ax1.plot(
    ranks[~mask_outlier],
    mod_zscore_data[~mask_outlier],
    "o",
    color="b",
    label="Cool Radiation",
)

# Anomaly data
ax1.plot(
    ranks[mask_outlier],
    mod_zscore_data[mask_outlier],
    "o",
    color="r",
    label="Anomalies",
)
ax1.axhline(threshold_radiation, color="r", label="threshold", alpha=0.5)

ax1.set_title("Modified Z-Score vs. Radiation", fontweight="bold")
ax1.set_xlabel("Radiation")
ax1.set_ylabel("Modified Z-Score")

ax1.legend(loc="best")

st.pyplot(fig)

with st.expander(label="Anomalias detectadas", expanded=True):
    st.write("Z-Score")
    st.dataframe(
        weather_df[weather_df["zscore_radiation"] < threshold_radiation][
            ["ano", "mes", "dia", "radiacao", "zscore_radiation"]
        ]
    )

    st.write("Modified Z-Score")
    st.dataframe(
        weather_df[weather_df["mod_zscore_radiation"] < threshold_radiation][
            ["ano", "mes", "dia", "radiacao", "mod_zscore_radiation"]
        ]
    )
