import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

def create_embeddings_and_store(docs):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db

st.title('🎬 Film Bot Assistant')
main_placeholder = st.empty()

st.sidebar.title('🎞️ Movie Article URLs')
urls = []
for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}')
    if url:
        urls.append(url)

if 'db' not in st.session_state:
    st.session_state.db = None

process_btn = st.sidebar.button('📥 Load Movie Info')

if process_btn and urls:
    from langchain_community.document_loaders import WebBaseLoader
    main_placeholder.write("🔄 Fetching and processing movie articles...")
    loader = WebBaseLoader(urls)
    documents = loader.load()

    from langchain.text_splitter import RecursiveCharacterTextSplitter


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)

    main_placeholder.write(f"✅ Loaded {len(docs)} chunks of movie info.")
    main_placeholder.write("🧠 Generating movie embeddings...")

    st.session_state.db = create_embeddings_and_store(docs)
    main_placeholder.write("✅ Vector store created for movie data.")

query = main_placeholder.text_input('💬 Ask something about the movies or actors:')

if query and st.session_state.db:
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA

    llm = ChatGroq(
        api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=0.7
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.db.as_retriever(),
        chain_type="stuff",
        return_source_documents=True
    )

    response = qa_chain({'query': query})
    st.header('🎯 Answer')
    st.write(response['result'])

    st.header('🔗 Sources')
    for i, doc in enumerate(response['source_documents']):
        source = doc.metadata.get('source', 'Unknown Source')
        st.markdown(f"**Source {i+1}:** {source}")
        st.markdown(f"📝 _Snippet:_ {doc.page_content[:300]}...")
