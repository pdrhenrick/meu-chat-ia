# frontend/app.py - CÓDIGO FINAL CORRIGIDO

import streamlit as st
import requests
import json
import os # <-- IMPORTAÇÃO CORRETA

# LÓGICA CORRETA PARA A URL DO BACKEND
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/chat")

st.set_page_config(page_title="Chat com IA", page_icon="🤖")
st.title("🤖 Chat com IA Personalizada")
st.caption("Faça uma pergunta sobre os dados que eu aprendi.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Qual a sua pergunta?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Pensando...")

        try:
            payload = {"question": prompt}
            response = requests.post(BACKEND_URL, json=payload)
            response.raise_for_status()

            full_response = response.json().get("answer", "Não recebi uma resposta válida.")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao conectar com o serviço de IA: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Erro de conexão: {e}"})
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Erro: {e}"})