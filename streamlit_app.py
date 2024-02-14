import streamlit as st
import os
import openai

from retriever import SparseRetriever
from prompt import PROMPT_1

def main():
    st.set_page_config(
    page_title="History Teller",
    page_icon=":books:")

    st.title("한국사 챗봇 History Teller")

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("실행")
    if process:
        if not openai_api_key:
            st.info("OpenAI API 키를 입력하세요.")
            st.stop()
        with st.spinner(text='데이터 수집중...'):
            data = load_data()
        with st.spinner(text='텍스트 토큰화중...'):
            st.session_state.retriever = SparseRetriever(data)
        st.write('한국사에 대해 질문해주세요.')

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
            with st.spinner("답변 작성중..."):
                response = text_generator(st.session_state.messages, openai_api_key)
                st.session_state.messages[-1]["content"] = query
                st.markdown(response)

# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def load_data():
    chunks = list()
    n = len(os.listdir('data/'))
    my_bar = st.progress(0., text=f"0 / {n}")
    for i, chunk in enumerate(os.listdir('data/')):
        if chunk.endswith('.txt'):
            with open(os.path.join('data/', chunk)) as f:
                content = f.read()
            chunks.append(content)
        my_bar.progress((i+1)/n, text=f'{i+1} / {n}')
    my_bar.empty()
    return chunks


def text_generator(messages, openai_api_key, model="gpt-4-turbo-preview", temperature=0):
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    answer = response.choices[0].message.content
    return answer

if __name__ == '__main__':
    main()
