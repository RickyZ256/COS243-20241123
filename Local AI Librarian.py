from llama_index import SimpleDirectoryReader, Document
import os
from EbookLib import epub
from pdfminer.high_level import extract_text
from Embedding_Generation import EmbeddingGenerator, VectorIndex

def load_ebooks(directory_path):
    documents = []
    
    # Iterate through files in the directory
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        
        # Process EPUB files
        if file_name.endswith('.epub'):
            try:
                print(f"Processing EPUB: {file_name}")
                text = extract_text_from_epub(file_path)
                documents.append(Document(text=text, metadata={"file_name": file_name}))
            except Exception as e:
                print(f"Error processing EPUB {file_name}: {e}")
        
        # Process PDF files
        elif file_name.endswith('.pdf'):
            try:
                print(f"Processing PDF: {file_name}")
                text = extract_text_from_pdf(file_path)
                documents.append(Document(text=text, metadata={"file_name": file_name}))
            except Exception as e:
                print(f"Error processing PDF {file_name}: {e}")
        
        # Skip unsupported files
        else:
            print(f"Skipping unsupported file: {file_name}")
    
    return documents

def extract_text_from_epub(file_path):
    book = epub.read_epub(file_path)
    text = []
    
    for item in book.items:
        if item.get_type() == epub.EpubHtml:
            text.append(item.get_content().decode('utf-8'))
    
    return "\n".join(text)

def extract_text_from_pdf(file_path):
    return extract_text(file_path)

def chunk_documents(documents, chunk_size=500):
    chunked_documents = []
    
    for doc in documents:
        words = doc.text.split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_id"] = i // chunk_size
            chunked_documents.append(Document(text=chunk_text, metadata=chunk_metadata))
    
    return chunked_documents


if __name__ == "__main__":
    # Load and process documents
    directory_path = "./ebooks"
    documents = load_ebooks(directory_path)
    chunked_documents = chunk_documents(documents)

    print(f"Processed {len(documents)} documents into {len(chunked_documents)} chunks.")

    # Generate embeddings
    chunked_texts = [doc.text for doc in chunked_documents]
    chunked_metadata = [doc.metadata for doc in chunked_documents]

    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(chunked_texts)

    # Build vector index
    vector_index = VectorIndex()
    vector_index.add_to_index(embeddings, chunked_metadata)
    vector_index.build_index()

    # Save the index
    vector_index.save_index("vector_index.pkl")
    print("Embedding and index generation complete.")