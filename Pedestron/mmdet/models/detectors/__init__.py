from .base import BaseDetector
from .single_stage import SingleStageDetector
# from .mgan import MGAN
from .csp import CSP
from .csp_blockcopy import CSPBlockCopy

__all__ = [
    'BaseDetector', 'SingleStageDetector', 'CSP', 'CSPBlockCopy'
]
