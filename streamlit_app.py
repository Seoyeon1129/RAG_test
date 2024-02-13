import streamlit as st
import os
import openai

from retriever import SparseRetriever
from prompt import PROMPT_1

def main():
    st.set_page_config(
    page_title="History Teller",
    page_icon=":books:")

    st.title("_History Teller :red[한국사 질문 응답 서비스]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        with st.spinner(text='데이터 수집중...'):
            data = load_data()
        with st.spinner(text='텍스트 토큰화중...'):
            st.session_state.retriever = SparseRetriever(data)
        st.write('한국사에 대해 질문해주세요.')
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] =[{"role": "system", "content": "한국사 관련 질문이 아닐 경우 답변을 거부해."},
                {"role": "system", "content": "질문에 오류 혹은 잘못된 정보가 있는지 확인하고, 있다면 이것을 지적하고 수정해."},
               ]

    for message in st.session_state.messages:
        if message["role"] != "system": 
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        contexts = st.session_state.retriever.retrieve(query)
        prompt = PROMPT_1.format(query=query, contexts=contexts)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                response = text_generator(st.session_state.messages, openai_api_key)
                st.markdown(response)

# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def load_data():
    chunks = list()
    for chunk in os.listdir('data/'):
        if chunk.endswith('.txt'):
            with open(os.path.join('data/', chunk)) as f:
                content = f.read()
            chunks.append(content)

    return chunks


def text_generator(messages, openai_api_key, model="gpt-4", temperature=0):
    openai.api_key = openai_api_key
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    answer = response.choices[0].message.content
    return answer

if __name__ == '__main__':
    main()
