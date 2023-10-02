import numpy as np
import pandas as pd
import streamlit as st
from sklearn.svm import OneClassSVM

from utils.info_messages import get_key_error_message_info


def svm_anomaly_scores(obs):
    oc_svm = OneClassSVM(gamma="auto").fit(obs)
    scores = oc_svm.decision_function(obs).flatten()

    # Find the largest score and use it to normalize the scores
    max_score = np.max(np.abs(scores))

    # scores from oc_svm use "negative is anomaly"
    # To follow our previous convention
    # we multiply by -1 and divide by the maximum score to get scores
    # in the range [-1, 1] with positive values indicating anomalies
    return -scores / max_score


st.sidebar.info(get_key_error_message_info())

st.title(
    "Exercício 3 - Aplicar SVM nos dados de meteorologia e imprimir as top 5 anomalias"
)
st.subheader("Visualização do Dataset")

if "weather_df_exercise3" not in st.session_state:
    weather_df = pd.read_csv("datasets/mediadiaria.csv")
    st.session_state["weather_df_exercise3"] = weather_df

weather_df = st.session_state.get("weather_df_exercise3")
st.dataframe(weather_df)

st.subheader("Adicionando Temperaturas Anômalas ao Dataset")
st.code(
    """
    anomaly_temps = [87.5, 103.9, 54.1, 27.431610, 27.431610]
    anomaly_years = [2011, 2015, 2016, 2017, 2018]
    for index, temp in enumerate(anomaly_temps):
        weather_df.loc[index, ["ano", "temp"]] = [anomaly_years[index], temp]

    weather_df.head()
"""
)

anomaly_temps = [87.5, 103.9, 54.1, 27.431610, 27.431610]
anomaly_years = [2011, 2015, 2016, 2017, 2018]
for index, temp in enumerate(anomaly_temps):
    weather_df.loc[index, ["ano", "temp"]] = [anomaly_years[index], temp]

st.dataframe(weather_df.head())

st.subheader("Conversão de Datas")
st.code(
    """
datetime_series = pd.to_datetime(
    weather_df[["ano", "mes", "dia"]].rename(columns={"ano": "year", "mes": "month", "dia": "day"})
)

weather_df["datetime"] = datetime_series
weather_df.head()
"""
)

datetime_series = pd.to_datetime(
    weather_df[["ano", "mes", "dia"]].rename(
        columns={"ano": "year", "mes": "month", "dia": "day"}
    )
)

weather_df["datetime"] = datetime_series
st.dataframe(weather_df.head())

st.subheader("Agrupando temperaturas por ano")
st.code(
    """
year_grouper = pd.Grouper(key="datetime", freq="A")
weather_grouped_by_year_df = weather_df.groupby(year_grouper).max()

weather_grouped_by_year_df.head()
"""
)

year_grouper = pd.Grouper(key="datetime", freq="A")
weather_grouped_by_year_df = weather_df.groupby(year_grouper).max()

st.dataframe(weather_grouped_by_year_df.head())

st.subheader("""Visualizando as anomalias""")
st.code(
    """
temperature_series = weather_grouped_by_year_df[["ano", "temp"]].sort_values(by="ano")
temperature_series
"""
)

temperature_series = weather_grouped_by_year_df[["ano", "temp"]].sort_values(by="ano")
st.dataframe(temperature_series)

st.code(
    """
temperature_array = np.array(temperature_series)
top_n_anomalies = svm_anomaly_scores(temperature_array).argsort()[:top_anomalies]

st.write(f"Top {top_anomalies} Anomalias")

for anomaly in top_n_anomalies:
    st.write(f"Ano: {int(temperature_array[anomaly][0])}, Temperatura: {round(temperature_array[anomaly][1], 2)}")
"""
)

top_anomalies = st.number_input(
    "Quantidade de Anomalias", value=3, step=1, min_value=1, max_value=5
)
temperature_array = np.array(temperature_series)
top_n_anomalies = svm_anomaly_scores(temperature_array).argsort()[:top_anomalies]

st.write(f"Top {top_anomalies} Anomalias")

for anomaly in top_n_anomalies:
    st.write(
        f"Ano: {int(temperature_array[anomaly][0])}, Temperatura: {round(temperature_array[anomaly][1], 2)}"
    )
