import fitz  # PyMuPDF

from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text


# Trace the exact location of the file
pdf_path = r"C:\Users\HP\Downloads\NIST.SP.800-53r5.pdf"
raw_text = extract_text_from_pdf(pdf_path)

# Split the text into chunks for later steps


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_text(raw_text)

print(f"Loaded and split {len(texts)} chunks.")



# Use a free model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example usage
text = "The NIST Cybersecurity Framework helps organizations manage risk."
vector = embeddings.embed_query(text)

print(vector[:5])  # Just show first 5 values of the embedding



# Create FAISS vector store
documents = [Document(page_content=t) for t in texts]  # 'texts' from your PDF
db = FAISS.from_documents(documents, embeddings)

# Example retrieval
query = "What are the 5 functions of the NIST framework?"
docs = db.similarity_search(query, k=3)

for i, doc in enumerate(docs, 1):
    print(f"\nDocument {i}:\n{doc.page_content}")
