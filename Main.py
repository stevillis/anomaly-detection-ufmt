import streamlit as st

from utils.info_messages import get_key_error_message_info

st.sidebar.info(get_key_error_message_info())

st.title("Pós Graduação em Gestão e Ciência de Dados - UFMT")
st.header("Detecção de Anomalias")

st.write(
    "Use o **Menu Lateral** para acessar os Exercícios de Métodos de Detecção de Anomalias utilizando Python."
)
