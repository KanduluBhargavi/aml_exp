
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Step 1: Load PDF
PDF_PATH = "telecom_guide.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"Loaded {len(pages)} pages from the PDF")

print("\nFirst 500 characters:")
print(pages[0].page_content[:500])

# Step 2: Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(pages)

print(f"\nTotal chunks: {len(chunks)}")
print(
    f"Average chunk length: "
    f"{sum(len(c.page_content) for c in chunks) // len(chunks)} characters"
)

print("\nExample chunk:")
print(chunks[1].page_content if len(chunks) > 1 else chunks[0].page_content)

# Step 3: Load embedding model
print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Store embeddings in Chroma
print("\nEmbedding chunks and storing in Chroma...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

print(
    f"Vector store ready. "
    f"{vector_store._collection.count()} vectors stored."
)

# Step 5: Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)

# Step 6: Test query
test_query = "What is VoLTE and how does it improve call quality?"

retrieved = retriever.invoke(test_query)

print(f"\nQuery: {test_query}")
print(f"Retrieved {len(retrieved)} chunks")

for i, doc in enumerate(retrieved, 1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content[:300])
