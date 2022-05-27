from dataclasses import dataclass
import torch
from transformers.file_utils import ModelOutput
from typing import Any, Optional, Tuple, List, Text
from src.recom_search.model.beam_node import BeamNode


@dataclass
class SearchModelOutput(ModelOutput):
    """
    """
    ends: List[BeamNode] = None
    output: Optional[str] = None
    output_token: Optional[Tuple[int]] = None
    score: Optional[Tuple[Tuple[float]]] = None
    score_avg: Optional[Tuple[Tuple[float]]] = None
    doc_id: Optional[str] = None
    reference: Optional[str] = None
    document: Optional[str] = None
    args: Optional[Any] = None
