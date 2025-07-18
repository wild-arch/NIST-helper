import fitz  # PyMuPDF – For extracting text from PDFs

import sentence_transformers

from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


# Step 1: Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text


# Path to the NIST PDF (Update this if needed)
pdf_path = "static/NIST.SP.800-53r5.pdf"
raw_text = extract_text_from_pdf(pdf_path)

# Step 2: Split PDF text into manageable chunks
text_splitter = CharacterTextSplitter(
    separator="\n",  # Split by newlines
    chunk_size=1000,  # Max chunk size
    chunk_overlap=200  # Overlap to maintain context
)
texts = text_splitter.split_text(raw_text)
print(f"Loaded and split {len(texts)} chunks.")

# Step 3: Convert text chunks into Document objects for vector storage
documents = [Document(page_content=t) for t in texts]

# Step 4: Load embeddings model from HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Create FAISS vectorstore from documents
db = FAISS.from_documents(documents, embeddings)

# Step 6: Load Ollama LLM (local model like mistral, codellama, etc.)
llm = Ollama(model="qwen3:8b")

#Step 7: Define your own prompt (this is where you learn!)
# Use {context} and {question} as placeholders
prompt = PromptTemplate(
    input_variables=["context", "question"],

template = """
You are a cybersecurity expert specialized in the NIST.SP.800-53r5 framework. 

Your role is to help users understand and apply NIST controls to real-world cybersecurity problems.

Your task:
1. Read the user’s question.
2. If the question is related to NIST.SP.800-53r5, provide a clear, structured answer using only the provided context.
3. If not related, reply: “The question is outside my scope.”
4. When answering, clearly restate the user’s question first.
5. If relevant, list the specific NIST controls involved, using bullet points where appropriate.

Context:
{context}

User Question:
{question}

Answer:
"""
)


# Step 8: Create the RetrievalQA chain with your custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff",  # Simple chain that passes all docs as one block
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# Step 9: Ask a question
print("💬 Cybersecurity Chatbot (type 'exit' to quit)")
print("-" * 50)

while True:
    # Ask user for input
    question = input("You: ")

    # Exit condition
    if question.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break

    try:
        # Run the chain and store the result
        response = qa_chain.invoke({"query": question})

        # Display the response
        print("\n🤖 Bot:\n", response['result'])

        # Print sources (optional)
        for i, doc in enumerate(response['source_documents'], 1):
            print(f"\n📄 Source {i}:\n{doc.page_content}")

    except Exception as e:
        print(f"[❌ ERROR] Could not get a response: {e}")


# Step 12: Example manual similarity search (optional test)
# query = "What are the 5 functions of the NIST framework?"
# docs = db.similarity_search(query, k=3)
#
# for i, doc in enumerate(docs, 1):
#     print(f"\nTop {i} similar doc:\n{doc.page_content}")
