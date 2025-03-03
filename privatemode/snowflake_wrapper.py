from typing import List, Optional

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings


class SnowflakeEmbedding(HuggingFaceEndpointEmbeddings):
    """
    Wrapper around HuggingFaceEndpointEmbeddings to implement a custom embed_query.
    """

    # def embed_query(self, text: str) -> List[float]:
    #     original_model_kwargs = self.model_kwargs or {}
    #     self.model_kwargs = {"prompt_name": "query"}
    #     try:
    #         embedding = super().embed_query(text)
    #     finally:
    #         self.model_kwargs = original_model_kwargs
    #     return embedding

    # async def aembed_query(self, text: str) -> List[float]:
    #     original_model_kwargs = self.model_kwargs or {}
    #     self.model_kwargs = {"prompt_name": "query"}
    #     try:
    #         embedding = await super().aembed_query(text)
    #     finally:
    #         self.model_kwargs = original_model_kwargs
    #     return embedding