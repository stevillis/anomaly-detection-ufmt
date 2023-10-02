import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import zscore

from utils.info_messages import get_key_error_message_info
from utils.stats import modified_zscore


def apply_zscore_analisys(dataframe):
    zscore_participation_rate = zscore(dataframe["Participation Rate"], ddof=0)
    dataframe = dataframe.assign(zscore=zscore_participation_rate)
    return dataframe


st.sidebar.info(get_key_error_message_info())

st.title("Exercício 1 - Modified Z-Score")
st.header("Análise de taxa de Participação no SAT")
st.subheader("Visualização do Dataset")

if "sat_ct_df" not in st.session_state:
    sat_ct_df = pd.read_csv("datasets/SAT_CT_District_Participation_2012.csv")
    st.session_state["sat_ct_df"] = sat_ct_df

sat_ct_df = st.session_state.get("sat_ct_df")

st.dataframe(sat_ct_df)


st.subheader("A. Análise das Anomalias com Modified Z-Score")
st.write("Cálculo do Modified Z-Score")
st.code(
    """
    mod_zscore_participation_rate, mad_goals = modified_zscore(sat_ct_df["Participation Rate"])
    sat_ct_df = sat_ct_df.assign(mod_zscore=mod_zscore_participation_rate)
    sat_ct_df.head()
"""
)

mod_zscore_participation_rate, mad_goals = modified_zscore(
    sat_ct_df["Participation Rate"]
)
sat_ct_df = sat_ct_df.assign(mod_zscore=mod_zscore_participation_rate)
st.dataframe(sat_ct_df.head())

sat_ct_df = apply_zscore_analisys(sat_ct_df)

st.write("Comparação entre Z-Score e Modified Z-Score")
fig, ax = plt.subplots(1, 1)

st.code(
    """
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))

# Z-Score
zscore_data = sat_ct_df["zscore"].copy().sort_values(ascending=False).values
ranks = np.linspace(1, len(zscore_data), len(zscore_data))
mask_outlier = zscore_data < threshold

ax0.plot(
    ranks[~mask_outlier],
    zscore_data[~mask_outlier],
    "o",
    color="b",
    label="OK schools",
)
ax0.plot(ranks[mask_outlier], zscore_data[mask_outlier], "o", color="r", label="Anomalies")
ax0.axhline(threshold, color="r", label="Threshold", alpha=0.5)
ax0.legend(loc="upper right")
ax0.set_title("Z-Score vs. School District", fontweight="bold")
ax0.set_xlabel("Ranked School District")
ax0.set_ylabel("Z-Score")

# Modified Z-Score
mod_zscore_data = sat_ct_df["mod_zscore"].copy().sort_values(ascending=False).values
ranks = np.linspace(1, len(mod_zscore_data), len(mod_zscore_data))
mask_outlier = mod_zscore_data < threshold

ax1.plot(
    ranks[~mask_outlier],
    mod_zscore_data[~mask_outlier],
    "o",
    color="b",
    label="OK schools",
)
ax1.plot(ranks[mask_outlier], mod_zscore_data[mask_outlier], "o", color="r", label="Anomalies")
ax1.axhline(threshold, color="r", label="Threshold", alpha=0.5)
ax1.legend(loc="upper right")
ax1.set_title("Modified Z-Score vs. School District", fontweight="bold")
ax1.set_xlabel("Ranked School District")
ax1.set_ylabel("Modified Z-Score")

plt.show()
"""
)

threshold = st.number_input("Valor do Threshold", value=-2.0, step=0.1, format="%.2f")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

# Z-Score
zscore_data = sat_ct_df["zscore"].copy().sort_values(ascending=False).values
ranks = np.linspace(1, len(zscore_data), len(zscore_data))
mask_outlier = zscore_data < threshold

ax0.plot(
    ranks[~mask_outlier],
    zscore_data[~mask_outlier],
    "o",
    color="b",
    label="OK schools",
)
ax0.plot(
    ranks[mask_outlier], zscore_data[mask_outlier], "o", color="r", label="Anomalies"
)
ax0.axhline(threshold, color="r", label="Threshold", alpha=0.5)
ax0.legend(loc="upper right")
ax0.set_title("Z-Score vs. School District", fontweight="bold")
ax0.set_xlabel("Ranked School District")
ax0.set_ylabel("Z-Score")

# Modified Z-Score
mod_zscore_data = sat_ct_df["mod_zscore"].copy().sort_values(ascending=False).values
ranks = np.linspace(1, len(mod_zscore_data), len(mod_zscore_data))
mask_outlier = mod_zscore_data < threshold

ax1.plot(
    ranks[~mask_outlier],
    mod_zscore_data[~mask_outlier],
    "o",
    color="b",
    label="OK schools",
)
ax1.plot(
    ranks[mask_outlier],
    mod_zscore_data[mask_outlier],
    "o",
    color="r",
    label="Anomalies",
)
ax1.axhline(threshold, color="r", label="Threshold", alpha=0.5)
ax1.legend(loc="upper right")
ax1.set_title("Modified Z-Score vs. School District", fontweight="bold")
ax1.set_xlabel("Ranked School District")
ax1.set_ylabel("Modified Z-Score")

st.pyplot(fig)

st.subheader("B. Você encontra as mesmas anomalias?")
st.markdown(
    """
**Resposta:**

Não. Usando o Threshold -2, além dos Distritos *New Britain*, *Windham*, *Eastern Connecticut Regional Educational Service* e *Stamford Academy*,
o Distrito de *New London* também foi considerado um outlier na Análise com Modified Z-Score.
"""
)

with st.expander(label="Anomalias detectadas", expanded=True):
    st.write("Z-Score")
    st.dataframe(
        sat_ct_df[sat_ct_df["zscore"] < threshold][
            ["District", "Participation Rate", "zscore"]
        ]
    )

    st.write("Modified Z-Score")
    st.dataframe(
        sat_ct_df[sat_ct_df["mod_zscore"] < threshold][
            ["District", "Participation Rate", "mod_zscore"]
        ]
    )


st.subheader("C. Discuta as descobertas")
st.markdown(
    """
O Modified Z-Score é menos afetado por valores extremos porque ele usa a mediana e o desvio absoluto mediano.

Isso significa que ele pode identificar pontos como anomalias que não seria identificados pelo Z-Score.
Esta é a razão para New London ter sido classificado como anomalia ao usar o Modified Z-Score.
"""
)
