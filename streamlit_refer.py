import streamlit as st
import chardet
import os
import openai
from loguru import logger
from konlpy.tag import Okt
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from retriever import SparseRetriever
from prompt import PROMPT_1

def main():
    st.set_page_config(
    page_title="History Teller",
    page_icon=":books:")

    st.title("_Private Data :red[QA Chat]_ :books:")

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
            retriever = SparseRetriever(data)
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] =[{"role": "system", "content": "한국사 관련 질문이 아닐 경우 답변을 거부해."},
                {"role": "system", "content": "질문에 오류 혹은 잘못된 정보가 있는지 확인하고, 있다면 이것을 지적하고 수정해."},
               ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        contexts = retriever.retrieve(query)
        prompt = PROMPT_1.format(query=query, contexts=contexts)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                response = text_generator(messages, openai_api_key)
                st.markdown(response)

# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def load_data_md():
    loader = UnstructuredMarkdownLoader("./data/kohist.md", mode='elements')
    data_md = loader.load()
    chunks = []
    for document in data_md:
        if len(document.page_content)>30:
            chunks.append(document)
    return chunks

def load_data_2007():
    with open("./data/kohist_2007.txt", "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]
    with open("./data/kohist_2007.txt", encoding=encoding) as f:
        data_2007 = f.read()


    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.create_documents([data_2007])
    chunks = []
    
    for document in documents:
        chunks.append(document)
        
    return chunks

def load_data_pdf(chunk_size=700, chunk_overlap = 100):
    loaders = []
    for pdf in os.listdir("./data/pdf"):
        loaders.append(PDFPlumberLoader(os.path.join("./data/pdf", pdf)))
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = []
    for loader in tqdm(loaders):
        chunks.extend(loader.load_and_split(text_splitter=text_splitter))
        
    return chunks                   

def load_data():
    chunks_md = load_data_md()
    chunks_07 = load_data_2007()
    # chunks_pdf = load_data_pdf()
    chunks_total = chunks_md + chunks_07 # + chunks_pdf
    filtered_chunks = filter_complex_metadata(chunks_total)
    filtered_texts = [doc.page_content for doc in filtered_chunks]
    return filtered_texts

def text_generator(messages, openai_api_key, model="gpt-4", temperature=0):
    openai.api_key = openai_api_key
    client = openai.OpenAI()
    messages = [{"role": "system", "content": "한국사 관련 질문이 아닐 경우 답변을 거부해."},
                {"role": "system", "content": "질문에 오류 혹은 잘못된 정보가 있는지 확인하고, 있다면 이것을 지적하고 수정해."},
                {"role": "user", "content": prompt},
               ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    answer = response.choices[0].message.content
    return answer

if __name__ == '__main__':
    main()
