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
#     "ragas>=0.1.3",
#     "datasets>=2.14.0",
#     "pandas>=2.0.0",
#     "langchain-anthropic>=0.1.1",
#     "langchain_openai",
#     "langchain_nvidia_ai_endpoints>=0.3.7",
#     "langchain_huggingface",
#     "langchain_qdrant"
# ]
# ///

# with reranker: https://app.ragas.io/dashboard/alignment/evaluation/e58a9578-d1f6-44e4-9a8d-b08f8bc3fb59 .. better results

"""
Evaluate RAG pipeline using RAGAS metrics.
This script evaluates the RAG pipeline in main.py against golden answers in questions_eu.json.
Run with: uv run evaluate_rag.py
"""

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from qdrant_client import QdrantClient
from snowflake_wrapper import SnowflakeEmbedding


# RAGAS imports
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper

# Import the RAG class from the new rag_pipeline.py file
from rag_pipeline import RAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
PM_API_KEY = os.environ.get("PM_API_KEY") or "bb79ba96-8a99-4faa-a673-3e029aba4100"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
PDF_PATH = os.path.join(os.path.dirname(__file__), "eu.pdf")
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions_eu.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "ragas_evaluation_results.csv")


def load_questions(file_path: str) -> List[Dict[str, Any]]:
    """Load questions and golden answers from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def setup_remote_embedder():
    """Set up the remote embedder similar to evaluate_rag.py."""
    return SnowflakeEmbedding(model="http://51.159.74.33:80")

def setup_rag_pipeline():
    """
    Set up the RAG pipeline similar to main.py.
    
    This function sets up the RAG pipeline with the following components:
    - A ChatOpenAI LLM
    - An UnstructuredLoader for loading documents
    - A setup_remote_embedder for generating embeddings
    - A RecursiveCharacterTextSplitter for splitting text into chunks
    - A QdrantClient for storing the vector database
    - A RAG pipeline with the above components
    
    The function also ingests the documents from the PDF into the RAG pipeline.
    
    Returns:
        A RAG pipeline instance.
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model_name="latest",
        base_url="http://localhost:8080/v1",
        api_key=PM_API_KEY
    )
    
    # Initialize document loader
    loader = UnstructuredLoader(
        PDF_PATH,
        url="http://51.159.9.172:8000/general/v0/general",
        partition_via_api=True,
        strategy="fast"
    )
    
    document_embedder = setup_remote_embedder()
    # document_embedder = NVIDIAEmbeddings(model="nvidia/llama-3.2-nv-embedqa-1b-v2", truncate="END")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=510,
        chunk_overlap=100
    )
    
    # Initialize vector database
    vector_db = QdrantClient(":memory:")
    
    # Initialize RAG pipeline
    ranker = None #NVIDIARerank(model="nvidia/llama-3.2-nv-rerankqa-1b-v2", top_n=4, truncate="END")  
    rag = RAG(vector_db, llm, loader, text_splitter, ranker, document_embedder)
    
    # Ingest documents
    logger.info("Ingesting documents into the RAG pipeline...")
    rag.ingest_docs(PDF_PATH)
    
    return rag


def setup_evaluation_llm():
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not found in environment variables. Using default LLM.")
        return ChatOpenAI(
            model_name="latest",
            base_url="http://localhost:8080/v1",
            api_key=PM_API_KEY
        )
    
    return ChatAnthropic(
        model="claude-3-5-haiku-latest",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0
    )

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_rag_response(rag: RAG, question: str) -> str:
    """Get response from RAG pipeline for a given question."""
    logger.info(f"Querying RAG pipeline with question: {question}")
    # task = 'Given a web search query, retrieve relevant passages that answer the query'
    # nquery = get_detailed_instruct(task_description=task, query=question)
    response_generator = rag.rag_chain(question, top_n=8)
    response = "".join(list(response_generator))
    return response


def get_contexts_for_question(rag: RAG, question: str) -> List[str]:
    """Get retrieved contexts for a question."""
    # This uses the internal retriever of the R AG pipeline
    top_k = 8  # Same as used in the evaluation
    retriever = rag.vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)
    return [doc.page_content for doc in docs]


def extract_ground_truth_contexts(question_data: Dict[str, Any]) -> List[str]:
    """Extract ground truth contexts from the Citation field."""
    if "Citation" not in question_data or not question_data["Citation"]:
        return []
    
    # Extract the cited_text from each citation
    return [citation["cited_text"] for citation in question_data["Citation"] if "cited_text" in citation]


def prepare_evaluation_dataset(questions: List[Dict[str, Any]], rag: RAG) -> EvaluationDataset:
    """Prepare dataset for RAGAS evaluation using the new format."""
    dataset = []
    
    for item in questions:
        question = item["Question"]
        reference = item["Answer"]
        
        # Get RAG response
        response = get_rag_response(rag, question)
        
        # Get contexts used for the answer
        retrieved_contexts = get_contexts_for_question(rag, question)
        
        dataset.append({
            "user_input": question,
            "retrieved_contexts": retrieved_contexts,
            "response": response,
            "reference": reference
        })
    
    # Create EvaluationDataset
    return EvaluationDataset.from_list(dataset)


def evaluate_rag_pipeline():
    """Evaluate RAG pipeline using RAGAS metrics."""
    logger.info("Starting RAG pipeline evaluation with RAGAS...")
    
    questions = load_questions(QUESTIONS_PATH)
    # questions = questions[:1] # TEST
    logger.info(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")
    
    # Setup RAG pipeline
    rag = setup_rag_pipeline()
    
    eval_llm = setup_evaluation_llm()
    evaluator_llm = LangchainLLMWrapper(eval_llm)

    # Prepare dataset for RAGAS
    logger.info("Preparing evaluation dataset...")
    evaluation_dataset = prepare_evaluation_dataset(questions, rag)
    
    # Define metrics for evaluation
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    # Run evaluation
    logger.info("Running RAGAS evaluation...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        experiment_name="NVIDIA RAG without reranker"
    )
    # Convert results to DataFrame for better visualization
    try:
        results_df = results.to_pandas()
        
        # Print results
        logger.info("\n=== RAGAS Evaluation Results ===")
        print(results_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']])
        # Calculate average scores
        avg_scores = results_df.mean(numeric_only=True)
        logger.info("\n=== Average Scores ===")
        for metric, score in avg_scores.items():
            print(f"{metric}: {score:.4f}")
        
        # Save results to CSV
        results_path = os.path.join(os.path.dirname(__file__), "ragas_evaluation_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")
    except Exception as e:
        logger.error(f"Error while converting results to DataFrame: {e}")

    # Upload results to RAGAS dashboard if token is available
    if os.environ.get("RAGAS_APP_TOKEN"):
        logger.info("Uploading results to RAGAS dashboard...")
        results.upload()
    else:
        logger.warning("RAGAS_APP_TOKEN not found in environment variables. Skipping upload to RAGAS dashboard.")
    
    return results_df


if __name__ == "__main__":
    evaluate_rag_pipeline()
