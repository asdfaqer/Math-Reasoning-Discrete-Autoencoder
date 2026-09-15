"""
Reasoning Compression Autoencoder - Top-level Model Interface.

Exports core model classes, vector quantizers, and evaluation probes.
"""

from cascading_memory_model import (
    VectorQuantizer,
    SoftHardVectorQuantizer,
    AccuracyProbe,
    VectorizedPositionPerceptron,
    TopMAttentionIndexer,
    CascadingMemoryEncoder,
    CascadingMemoryDecoder,
    CascadingMemoryAutoencoder,
    SmallCascadingMemoryAutoencoder,
)

# Compatibility aliases
ReasoningAutoencoder = CascadingMemoryAutoencoder
LatentDecoder = CascadingMemoryDecoder
AdaptiveReasoningAutoencoder = CascadingMemoryAutoencoder
LegacyReasoningAutoencoder = CascadingMemoryAutoencoder

__all__ = [
    "VectorQuantizer",
    "SoftHardVectorQuantizer",
    "AccuracyProbe",
    "VectorizedPositionPerceptron",
    "TopMAttentionIndexer",
    "CascadingMemoryEncoder",
    "CascadingMemoryDecoder",
    "CascadingMemoryAutoencoder",
    "SmallCascadingMemoryAutoencoder",
    "ReasoningAutoencoder",
    "LatentDecoder",
    "AdaptiveReasoningAutoencoder",
    "LegacyReasoningAutoencoder",
]
