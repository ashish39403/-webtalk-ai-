import streamlit as st
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from streamlit_chat import message
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever 
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain  


#Vector store
def get_vector_store(url):
    loader = WebBaseLoader(url)
    documents=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =1000,
        chunk_overlap = 100,
    )
    chunks= text_splitter.split_documents(documents)
    return chunks


#Embeddings
def get_embedded(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    vector_db = Chroma.from_documents(chunks , embedding=embeddings)
    return vector_db



def get_context_retriever_chain(vector_db):
    llm = ChatOllama(model="gemma3:1b", temperature=0.5)
    
    retriever = vector_db.as_retriever()
    prompt= ChatPromptTemplate.from_messages([
        
        MessagesPlaceholder(variable_name='chat_history'),
        ("user","{input}"),
        ("user", "Given the above conversation , generate a search query to look up in order to get information relevant to the conversation"),
      
    ])
    retriever_chain = create_history_aware_retriever(llm , retriever ,prompt)
    return retriever_chain
    
    

def get_stuff_chain(retriever_chain):
    llm = ChatOllama(model="gemma3:1b", temperature=0.5)
    prompt =ChatPromptTemplate.from_messages([
        ("system" ,"Answer the user question based on the context below \n {context}\n"),
        
        MessagesPlaceholder(variable_name="chat_history"),
        
        ("user", "{input}"),
    ])
    stuff_docs =create_stuff_documents_chain(llm ,prompt)
    full_chains = create_retrieval_chain(retriever_chain ,stuff_docs)
    return full_chains
    


def get_response(user_query):
    
    retriever_chain = get_context_retriever_chain(st.session_state.vector_db)
    conversaaiton_rag_chain = get_stuff_chain(retriever_chain)
    
    
    # response = get_response(user_input)
    response=conversaaiton_rag_chain.invoke({
        "chat_history":st.session_state.chat_history,
        "input":user_input
    })
    return response['answer']
    

#Page configuration section
st.set_page_config(page_icon='🤖', page_title='Chat With Website')
st.title('💬Chat With Websites....')

#Sidebar
with st.sidebar:
    st.title('Enter the URL of website')
    website_url= st.text_input('Put URL :-')
if website_url is None or website_url=="":
    st.info('Please Enter a Website URL')
else:
    

    #Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= [
        AIMessage(content="Hi sir how may I Help You")
        ]
    if "vector_db"not in st.session_state:
         chunks = get_vector_store(website_url)
         
         st.session_state.vector_db = get_embedded(chunks)
        
   
    #User input  
    user_input=st.chat_input('Ask Your Question Here')
    if user_input is not None and user_input!="":
        response = get_response(user_input)
        # st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    

        #For conversation(To display on the screen).
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage ):
                with st.chat_message('AI'):
                    st.write(message.content)
            elif isinstance(message ,HumanMessage):
                with st.chat_message('Human'):
                    st.write(message.content)