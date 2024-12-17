import os
import gradio as gr
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from Embedding_Generation import EmbeddingGenerator, VectorIndex
from EbookLib import epub
from pdfminer.high_level import extract_text


def load_ebooks(directory_path):
    documents = []
    
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        if file_name.endswith('.epub'):
            try:
                print(f"Processing EPUB: {file_name}")
                text = extract_text_from_epub(file_path)
                documents.append(Document(text=text, metadata={"file_name": file_name}))
            except Exception as e:
                print(f"Error processing EPUB {file_name}: {e}")
        

        elif file_name.endswith('.pdf'):
            try:
                print(f"Processing PDF: {file_name}")
                text = extract_text(file_path)
                documents.append(Document(text=text, metadata={"file_name": file_name}))
            except Exception as e:
                print(f"Error processing PDF {file_name}: {e}")
    
    return documents

def extract_text_from_epub(file_path):
    book = epub.read_epub(file_path)
    text = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text += item.get_body_content().decode('utf-8')
    return text

def load_and_embed_documents(directory_path):
    documents = load_ebooks(directory_path)
    texts = [doc.text for doc in documents]
    
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(texts)

    vector_index = VectorIndex()
    vector_index.add_to_index(embeddings, [doc.metadata for doc in documents])
    vector_index.build_index()
    
    return vector_index, documents

def search_books(query, vector_index, documents):

    embedding_generator = EmbeddingGenerator()
    query_embedding = embedding_generator.generate_embeddings([query])[0]
    

    results = vector_index.search(query_embedding)
    

    search_results = []
    for result in results:
        doc_metadata = documents[result['index']].metadata
        search_results.append(f"File: {doc_metadata['file_name']} - Score: {result['score']:.4f}")
    
    return "\n".join(search_results)

def run_gradio_interface(directory_path):
    vector_index, documents = load_and_embed_documents(directory_path)
    
    gr.Interface(
        fn=lambda query: search_books(query, vector_index, documents),
        inputs=gr.Textbox(label="Enter your search query"),
        outputs=gr.Textbox(label="Search Results", lines=10),
        title="Local AI Librarian: Intelligent Book Search",
        description="Search for books by querying the content of EPUB and PDF files."
    ).launch()

run_gradio_interface("/path/to/your/ebook/directory")