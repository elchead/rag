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
#     "unstructured[pdf]",
# ]
# ///

# Local Qdrant + Remote Unstructured API + Remote Embedding + Remote Reranker + Remote LLM
# PM_API_KEY=<key> uv run main.py

import logging
import os
from typing import List, Generator
from dotenv import load_dotenv

from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIARerank
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global models
document_embedder = NVIDIAEmbeddings(model="nvidia/llama-3.2-nv-embedqa-1b-v2", truncate="END")
ranker = NVIDIARerank(model="nvidia/llama-3.2-nv-rerankqa-1b-v2", top_n=4, truncate="END")
vector_db_top_k = int(os.environ.get("VECTOR_DB_TOPK", 40))
pm_api_key = os.environ.get("PM_API_KEY")

class QdrantRAG:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(url="http://localhost:6333")

        # Initialize LLM
        self.llm = ChatOpenAI(model_name="latest",base_url="http://localhost:8080/v1",api_key=pm_api_key)

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
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Get embedding dimension from the model
            embedding_size = len(document_embedder.embed_query("test"))
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
            embeddings=document_embedder
        )

    def ingest_docs(self, data_dir: str, filename: str) -> None:
        """Ingest documents into Qdrant vector store"""
        try:
            # Load documents
            file_path = os.path.join(data_dir, filename)
            logger.info(f"Loading document from {file_path}")
            raw_documents = UnstructuredLoader(file_path,url="http://localhost:8000/general/v0/general",partition_via_api=True).load()

            # Add documents to vector store
            self.vector_store.add_documents(raw_documents)
            logger.info(f"Successfully ingested document: {filename}")

        except Exception as e:
            logger.error(f"Failed to ingest document: {str(e)}")
            raise ValueError(f"Failed to upload document. {str(e)}")

    def rag_chain(self, query: str, chat_history: List = None, top_n: int = 4) -> Generator[str, None, None]:
        """Execute RAG chain to answer queries using the knowledge base"""
        try:
            # Set up retrieval parameters
            top_k = vector_db_top_k if ranker else top_n
            logger.info(f"Setting retriever top k as: {top_k}")

            # Create retriever
            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

            # Create prompt template
            prompt = ChatPromptTemplate.from_template(self.rag_template)

            if ranker:
                logger.info(
                    f"Narrowing the collection from {top_k} results and further narrowing it to "
                    f"{top_n} with the reranker for rag chain."
                )
                logger.info(f"Setting ranker top n as: {top_n}")
                ranker.top_n = top_n

                # Create reranker chain
                reranker = RunnableAssign({
                    "context": lambda input: ranker.compress_documents(
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

    def delete_docs(self, filenames: List[str]) -> bool:
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

def main():
    # Example usage
    rag = QdrantRAG()

    # Ingest a document
    print("INGESTING DOCUMENT")
    rag.ingest_docs("/Users/adria/Downloads", "AI Foundation Models Community License.pdf")
    print("INGESTED DOCUMENT")

    # Query the system
    response = rag.rag_chain("Can I use the NVIDIA AI Foundation Models Community for commercial purposes?")
    for chunk in response:
        print(chunk, end="")

if __name__ == "__main__":
    main()
