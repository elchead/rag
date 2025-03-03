#!/usr/bin/env python
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
#     "unstructured-client==0.29.0",
#     "langchain-qdrant"
# ]
# ///

"""
RAG Pipeline implementation for question answering over documents.
"""

import logging
import os
import time
from typing import List, Generator

from langchain.schema.embeddings import Embeddings
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from langchain_qdrant import QdrantVectorStore

from qdrant_client.models import Distance, VectorParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, vector_db, llm, loader, text_splitter, ranker, document_embedder):
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
        self.vector_store = QdrantVectorStore(
        client=self.client,
        collection_name=self.collection_name,
        embedding=self.document_embedder,
        )
        


    def ingest_docs(self, file_path: str) -> None:
        """Ingest documents into Qdrant vector store"""
        try:
            # Load documents
            logger.info(f"Loading document from {file_path}")
            time_start = time.time()
            raw_documents = self.loader.load()
            time_end = time.time()
            logger.info(f"Loading document from {file_path} took {time_end - time_start} seconds")
            # Split documents
            logger.info("Splitting documents with recursive character splitter")
            split_docs = self.text_splitter.split_documents(raw_documents)
            logger.info(f"Document splitting complete. Original: {len(raw_documents)}, "
                       f"After splitting: {len(split_docs)}")
            # print("\n=== Document Chunks after Text Splitter ===")
            # for i, doc in enumerate(split_docs):
            #     print(f"Chunk {i+1} length: {len(doc.page_content)} chars")
            #     print(f"Content: {doc.page_content}")
            #     print("----")

            # Add documents to vector store
            time_start = time.time()
            self.vector_store.add_documents(split_docs, batch_size=32) # required by HF TEI
            time_end = time.time()
            logger.info(f"Successfully ingested documents: {file_path} took {time_end - time_start} seconds")

        except Exception as e:
            logger.error(f"Failed to ingest document: {str(e)}")
            raise ValueError(f"Failed to upload document. {str(e)}")

    def rag_chain(self, query: str, chat_history: List = None, top_n: int = 8) -> Generator[str, None, None]:
        """Execute RAG chain to answer queries using the knowledge base"""
        try:
            # Set up retrieval parameters
            vector_db_top_k = int(os.environ.get("VECTOR_DB_TOPK", 40))
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

                # Create RAG chain with reranking
                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | reranker
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
            else:
                # Create RAG chain without reranking
                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )

            # Execute chain
            logger.info(f"Executing RAG chain for query: {query}")
            time_start = time.time()
            for chunk in rag_chain.stream(query):
                yield chunk
            time_end = time.time()
            logger.info(f"RAG chain execution completed in {time_end - time_start} seconds")

        except Exception as e:
            logger.error(f"Failed to execute RAG chain: {str(e)}")
            yield f"I'm sorry, I encountered an error while trying to answer your question: {str(e)}"


# Helper functions
def load_pdf_content(loader):
    """Load PDF content directly as a string."""
    raw_documents = loader.load()
    return "\n\n".join([doc.page_content for doc in raw_documents])


def direct_query(llm, content, query):
    """Execute direct query with full content in context."""
    template = """You are a helpful AI assistant. Use the following document to answer the question.
    If you don't know the answer, just say you don't know.

    Document: {content}

    Question: {query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({"content": content, "query": query})
