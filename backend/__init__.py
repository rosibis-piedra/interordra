"""
InterOrdra - Framework para detección de desacoplamientos semánticos
Detecta gaps en comunicación entre sistemas diversos
"""

__version__ = "0.1.0"
__author__ = "Rosibis"

from .embeddings import SemanticEmbedder
from .clustering import ConceptClusterer
from .gap_detector import GapDetector
from .analyzer import InterOrdraAnalyzer

__all__ = [
    'SemanticEmbedder',
    'ConceptClusterer',
    'GapDetector',
    'InterOrdraAnalyzer'
]