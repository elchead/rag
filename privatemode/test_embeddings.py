# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "langchain>=0.1.0",
#     "langchain-community>=0.0.20",  # Updated to newer version
#     "langchain-core>=0.1.0",
#     "requests>=2.31.0"  # Added for direct API calls
# ]
# ///

import logging
import requests
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url.rstrip("/")

    def embed_query(self, text: str) -> List[float]:
        """Get embeddings for a single text."""
        response = requests.post(
            f"{self.base_url}/embed",
            json={"inputs": [text]}
        )
        response.raise_for_status()
        embeddings = response.json()
        return embeddings[0]

def get_embedding():
    # Initialize the embedding model
    embeddings = EmbeddingClient()

    # Sample text to embed
    text = "This is a sample text to test embeddings."

    # Get embedding
    try:
        logger.info("Getting embedding for text: %s", text)
        embedding = embeddings.embed_query(text)
        logger.info("Embedding dimension: %d", len(embedding))
        logger.info("First few values: %s", embedding[:5])
        return embedding
    except Exception as e:
        logger.error("Failed to get embedding: %s", str(e))
        raise

def main():
    embedding = get_embedding()
    print(f"\nSuccessfully generated embedding with dimension {len(embedding)}")

if __name__ == "__main__":
    main()
