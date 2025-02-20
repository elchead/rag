# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "qdrant-client>=1.13.0",
#     "langchain>=0.1.0",
#     "langchain-openai>=0.0.2",
#     "langchain-unstructured",
#     "langchain-community>=0.3.7",
#     "langchain-nvidia-ai-endpoints>=0.3.7",
#     "python-dotenv>=1.0.0",
#     "requests>=2.31.0",
#     "unstructured-client==0.29.0"
# ]
# ///

# Local Qdrant + Remote Unstructured API + Remote Embedding + Remote Reranker + Remote LLM
# PM_API_KEY=<key> uv run main.py

import logging
import os
from typing import List, Generator
from dotenv import load_dotenv
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
vector_db_top_k = int(os.environ.get("VECTOR_DB_TOPK", 40))
pm_api_key = os.environ.get("PM_API_KEY") or "bb79ba96-8a99-4faa-a673-3e029aba4100"

class RAG:
    def __init__(self,vector_db, llm, loader, text_splitter, ranker, document_embedder):
        self.client = vector_db
        self.llm = llm
        self.loader = loader
        self.text_splitter = text_splitter
        self.ranker = ranker
        self.document_embedder = document_embedder
        # Initialize vector store
        self.collection_name = "documents"
        self._init_collection()

        # Initialize prompts
        self.rag_template = """You are a helpful AI assistant. Use the following context to answer the question.
        If you don't know the answer, just say you don't know.

        Context: {context}

        Question: {question}
        """

    def _init_collection(self):
        """Initialize Qdrant collection with proper configuration"""
        self.client.delete_collection(self.collection_name) # start from fresh collection every time
        #self.client.get_collection(self.collection_name)
        embedding_size = len(self.document_embedder.embed_query("test"))
        logger.info(f"Embedding size: {embedding_size}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embedding_size,
                distance=Distance.COSINE
            ),
        )

        # Initialize Langchain's Qdrant wrapper
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.document_embedder
        )

    def ingest_docs(self, file_path: str) -> None:
        """Ingest documents into Qdrant vector store"""
        try:
            # Load documents
            logger.info(f"Loading document from {file_path}")
            raw_documents = self.loader.load()

            # Split documents
            logger.info("Splitting documents with recursive character splitter")
            split_docs = self.text_splitter.split_documents(raw_documents)
            logger.info(f"Document splitting complete. Original: {len(raw_documents)}, "
                       f"After splitting: {len(split_docs)}")

            # Add documents to vector store
            self.vector_store.add_documents(split_docs)
            logger.info(f"Successfully ingested document: {file_path}")

        except Exception as e:
            logger.error(f"Failed to ingest document: {str(e)}")
            raise ValueError(f"Failed to upload document. {str(e)}")

    def rag_chain(self, query: str, chat_history: List = None, top_n: int = 8) -> Generator[str, None, None]:
        """Execute RAG chain to answer queries using the knowledge base"""
        try:
            # Set up retrieval parameters
            top_k = vector_db_top_k if self.ranker else top_n
            logger.info(f"Setting retriever top k as: {top_k}")

            # Create retriever
            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

            # Create prompt template
            prompt = ChatPromptTemplate.from_template(self.rag_template)

            if self.ranker:
                logger.info(
                    f"Narrowing the collection from {top_k} results and further narrowing it to "
                    f"{top_n} with the reranker for rag chain."
                )
                logger.info(f"Setting ranker top n as: {top_n}")
                self.ranker.top_n = top_n

                # Create reranker chain
                reranker = RunnableAssign({
                    "context": lambda input: self.ranker.compress_documents(
                        query=input['question'],
                        documents=input['context']
                    )
                })

                # Create retriever chain with reranking
                retriever = {"context": retriever, "question": RunnablePassthrough()} | reranker
                docs = retriever.invoke(query)
                docs = [d.page_content for d in docs.get("context", [])]

                chain = prompt | self.llm | StrOutputParser()
            else:
                # Create simple retriever chain
                docs = retriever.invoke(query)
                docs = [d.page_content for d in docs]
                chain = prompt | self.llm | StrOutputParser()

            # Execute chain
            logger.info(f"Executing RAG chain for query: {query}")
            return chain.stream({"question": query, "context": docs})

        except Exception as e:
            logger.error(f"Failed to execute RAG chain: {str(e)}")
            return iter(["Failed to generate response. Please try again."])

    def delete_docs(self) -> bool:
        """Delete documents from vector store"""
        try:
            # Note: This is a simplified version. In a real implementation,
            # you'd want to track document IDs and delete specific points
            self.client.delete_collection(self.collection_name)
            self._init_collection()
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False

def load_pdf_content(loader: UnstructuredLoader) -> str:
    """Load PDF content directly as a string."""
    raw_documents = loader.load()
    return "\n".join([doc.page_content for doc in raw_documents])

def direct_query(llm: ChatOpenAI, content: str, query: str) -> Generator[str, None, None]:
    """Execute direct query with full content in context."""
    prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant. Use the following document to answer the question.
    If you don't know the answer, just say you don't know.

    Document: {content}

    Question: {question}
    """)

    chain = prompt | llm | StrOutputParser()
    return chain.stream({"content": content, "question": query})

def main():
    pdf_path = os.path.join(os.path.dirname(__file__), "AI Foundation Models License.pdf")
    query = "Under what conditions can NVIDIA AI Foundation Models be used for production without an NVIDIA AI Enterprise subscription?"
    llm = ChatOpenAI(
            model_name="latest",
            base_url="http://localhost:8080/v1",
            api_key=pm_api_key
    )
    loader = UnstructuredLoader(pdf_path,url="http://localhost:8000/general/v0/general",partition_via_api=True)
    print("Loaded loader","http://localhost:8000")
    document_embedder = OpenAIEmbeddings(
    base_url="http://localhost:9090/v1",
    api_key="not-needed",
    chunk_size=32
    ) # the small embedding model is not good enough for the query
    # document_embedder = NVIDIAEmbeddings(model="nvidia/llama-3.2-nv-embedqa-1b-v2", truncate="END")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50)
    ranker = None # without reranker, the results are not good enough for the query
    # ranker = NVIDIARerank(model="nvidia/llama-3.2-nv-rerankqa-1b-v2", top_n=4, truncate="END")


    # Ingest document for RAG
    print("\n=== RAG Approach ===")
    vector_db = QdrantClient(":memory:") #(path="./db") (for persistence)  # #QdrantClient(url="http://localhost:6333")
    rag = RAG(vector_db, llm, loader, text_splitter, ranker, document_embedder)
    rag_start = time.time()
    rag.ingest_docs(pdf_path)
    response = rag.rag_chain(query)
    for chunk in response:
        print(chunk, end="")
    rag_time = time.time() - rag_start
    print(f"\nRAG Time: {rag_time:.2f} seconds")

    # Time direct context approach
    print("\n\n=== Direct Context Approach ===")
    direct_start = time.time()
    full_content = load_pdf_content(loader)
    print("Loading time for PDF content: ", time.time() - direct_start)
    response = direct_query(rag.llm, full_content, query)
    for chunk in response:
        print(chunk, end="")
    direct_time = time.time() - direct_start
    print(f"\nDirect Context Time: {direct_time:.2f} seconds")

    # Print comparison
    print("\n\n=== Time Comparison ===")
    print(f"RAG Time: {rag_time:.2f} seconds")
    print(f"Direct Context Time: {direct_time:.2f} seconds")
    print(f"Difference: {abs(rag_time - direct_time):.2f} seconds")

if __name__ == "__main__":
    main()
