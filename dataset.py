"""
Dataset module proxy exposing DynamicReasoningTraceDataset and get_dataloaders from data.dataset.
"""
import os
import sys

# Ensure local data directory is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import DynamicReasoningTraceDataset, get_dataloaders

__all__ = ["DynamicReasoningTraceDataset", "get_dataloaders"]
