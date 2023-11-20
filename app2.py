import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from streamlit_extras.add_vertical_space import add_vertical_space


openai_api_key =st.secrets['OPENAI_API_KEY']


with st.sidebar:
    st.title('PDF Chat App')

    with st.sidebar:
        st.title('PDF Chat App')

        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.write("Chat History:")
        for message in st.session_state["messages"]:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with  st.chat_message("assistant"):
                    st.markdown(message["content"])
        add_vertical_space(5)
#front end  chat set up
st.title('Your Wikipedia friend')

url = st.text_input("Enter the URL")

st.header("Chat with URL Content 💬")



if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if url:
    try:
        #for pulling the information from web
        response = requests.get(url)#web requesting
        if response.status_code == 200:
            # extracting HTML content from response
            soup = BeautifulSoup(response.content, 'html.parser')
            page_content = soup.get_text()# Retrieves the text from the parsed HTML.



#Using Language Models for Chatbot Interaction
            embeddings = OpenAIEmbeddings()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text=page_content)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)#vector store for the text chunks using embeddings.

            query = st.text_input("Ask a question about the content from the URL")

            if query:
                chat_history = []
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                llm = ChatOpenAI()
                CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template("Your custom prompt template here.")
                #retrieval chain to generate a response to the user's query.
                conversation_chain = ConversationalRetrievalChain.from_llm(llm, VectorStore.as_retriever(),
                                                                           condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                                                           memory=memory)
#Generates a response based on the user's query using the language model and the content from the URL.
                response = conversation_chain({"question": query, "chat_history": chat_history})


#The assistant's response is then displayed in the chat interface using
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                chat_history.append((query, response))
# error handling using try and except
        else:
            st.error("Failed to fetch content from the URL. Please enter a valid URL.")
    except requests.RequestException as e:
        st.error(f"Request Error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.error("Please enter a valid URL.")
