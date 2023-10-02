import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from utils.info_messages import get_key_error_message_info


def plot_point_and_k_neighbors(X, highlight_index, n_neighbors=2):
    "Plota os pontos em X e mostra os n_vizinhos do ponto destacado"
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    dist, index = nn.kneighbors()

    src_pt = X[highlight_index, :]

    fig, ax = plt.subplots()

    # draw lines first, so points go on top
    for dest_index in index[highlight_index]:
        dest_pt = X[dest_index, :]
        ax.plot(*list(zip(src_pt, dest_pt)), "k--")
    ax.plot(X[:, 0], X[:, 1], "o", label="Não k-vizinhos", alpha=0.3)
    ax.plot(*src_pt, "o", label="O ponto buscado")
    ax.plot(
        X[index[highlight_index], 0],
        X[index[highlight_index], 1],
        "o",
        label="k-neighbors",
    )
    ax.set_xlabel("Temperatura")
    ax.set_ylabel("Radiação")
    ax.legend()

    st.pyplot(fig)


def nn_outlier_scores(obs, n_neighbors=1):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(obs)

    dists, idx = nn.kneighbors()
    scores = dists[:, -1]

    return scores


def print_ranked_scores(obs, scores, k_points):
    scores_and_obs = sorted(zip(scores, obs), key=lambda t: t[0], reverse=True)
    # print("Pontos(x, y)\t\tScore")
    # print("-" * 20)

    count = 0
    k_points_list = []
    for index, score_ob in enumerate(scores_and_obs):
        score, point = score_ob
        # print(f"{index + 1:3d}. {point}\t\t{score:6.4f}")

        if count < k_points:
            k_points_list.append(point)
        else:
            break

        count += 1

    return k_points_list


def plot_outliers(X, k_points):
    outliers = print_ranked_scores(X, nn_outlier_scores(X, k_points), k_points=k_points)

    fig, ax = plt.subplots()
    ax.plot(X[:, 0], X[:, 1], "o", label="Dados Normais", alpha=0.3)
    for outlier in outliers:
        ax.plot(outlier[0], outlier[1], "o", label="Outliers", alpha=0.3, color="r")
        # print(outlier)

    ax.set_xlabel("Temperatura")
    ax.set_ylabel("Radiação")

    ax.legend(["Dados Normais", "Outliers"])

    st.pyplot(fig)


def plt_anomalies_kmeans(data, n_anomalies):
    anomaly_idx = np.argsort(score)[::-1][:n_anomalies]
    anomaly_mask = np.zeros(len(data))
    anomaly_mask[anomaly_idx] = 1
    colors = ["blue", "orange", "green"]

    fig, ax = plt.subplots()

    for label, color in enumerate(colors):
        mask = (km.labels_ == label) & (anomaly_mask == 0)
        ax.plot(
            data[mask, 0],
            data[mask, 1],
            marker="o",
            linestyle="none",
            color=color,
            label=f"Cluster {label}",
        )
        ax.plot(*km.cluster_centers_[label], marker="x", color="k")

    ax.plot(
        data[anomaly_idx, 0],
        data[anomaly_idx, 1],
        marker="o",
        linestyle="none",
        color="r",
        label="Anomaly",
    )

    ax.text(165000, 19, s="x Cluster Center")

    ax.set_xlim(-100000, 250000)

    ax.legend(loc="lower right")

    st.pyplot(fig)


st.sidebar.info(get_key_error_message_info())

st.title(
    "Exercício 3 - Aplicar SVM nos dados de meteorologia e imprimir as top 5 anomalias"
)
st.subheader("Visualização do Dataset")

if "weather_df" not in st.session_state:
    weather_df = pd.read_csv("datasets/weather.csv")
    st.session_state["weather_df"] = weather_df

weather_df = st.session_state.get("weather_df")
st.dataframe(weather_df)

# Preprocessing
## Removing null values
cleaned_weather_df = weather_df.dropna()

## Removing out of bounds data
cleaned_weather_df = cleaned_weather_df[(cleaned_weather_df["T_z1"] <= 1000)]

st.subheader("Gráfico de Dispersão - Radiação vs Hora")
st.code(
    """
plt.title("Radiação vs Hora")
plt.scatter(
    cleaned_weather_df["Rn"],
    cleaned_weather_df["T_z1"],
)
"""
)

fig, ax = plt.subplots()
plt.title("Radiação vs Hora")
plt.scatter(
    cleaned_weather_df["Rn"],
    cleaned_weather_df["T_z1"],
)

st.pyplot(fig)

st.subheader("Aplicando KNN")
X = cleaned_weather_df[["Rn", "T_z1"]].to_numpy()

col1, col2 = st.columns(2)
with col1:
    point = st.number_input(
        label="Índice do Ponto", value=205, step=1, min_value=0, max_value=len(X) - 1
    )

with col2:
    neighbors = st.number_input(
        label="Quantidade de Vizinhos",
        value=10,
        step=1,
        min_value=0,
        max_value=len(X) - 1,
    )

plot_point_and_k_neighbors(X, point, neighbors)

st.markdown("##### Visualizando Outliers")
col_outliers, _, _ = st.columns(3)
with col_outliers:
    outliers = st.number_input(
        label="Número de Outliers", value=8, step=1, min_value=0, max_value=len(X) - 1
    )

plot_outliers(X, outliers)

st.subheader("KMeans")
st.markdown("##### Clusters")

col_n_clusters, _, _ = st.columns(3)
with col_n_clusters:
    n_clusters = st.number_input(
        label="Quantidade de Clusters", value=3, step=1, min_value=1, max_value=15
    )

km = KMeans(n_clusters=n_clusters, n_init="auto").fit(X)

fig, ax = plt.subplots()

for label in range(n_clusters):
    mask = km.labels_ == label
    # print('Mask: ',km.labels_)
    ax.plot(X[mask, 0], X[mask, 1], "o", label=f"Cluster {label}")

ax.legend()
st.pyplot(fig)


st.markdown("##### Detecção de Anomalias com 3 Clusters")

kmeans = KMeans(n_clusters=3, n_init="auto").fit(X)
centers = kmeans.cluster_centers_[kmeans.labels_]

# Obtem as distâncias aos centros e as usa como pontuações
score = np.linalg.norm(X - centers, axis=1)

col_n_anomalies, _, _ = st.columns(3)
with col_n_anomalies:
    n_anomalies = st.number_input(
        label="Quantidade de Anomalias",
        value=6,
        step=1,
        min_value=1,
        max_value=len(X) - 1,
    )

plt_anomalies_kmeans(X, n_anomalies)
