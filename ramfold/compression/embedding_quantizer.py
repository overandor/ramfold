"""
RAMFold Embedding Quantizer — C++ embedding vaultlet interface.

Quantize embeddings to 8/4-bit precision for memory savings.
"""

from .context_compressor import EmbeddingQuantizer

__all__ = ["EmbeddingQuantizer"]
