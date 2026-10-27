"""Installable inference API; importing it does not load a model or CUDA."""
from testtime.contracts import PublicRequest
from testtime.harness import Policy
from testtime.retrieval import Document

from .library import SkillLibrary
from .runtime import Harness, make_request, recommended_policy
from .context_agent import ContextAgent
from .memory import MemoryItem, MemoryStore, RefinementLoop, Verification
from .dataset_adapter import prepare_dataset_request

__version__ = '0.2.0'
__all__ = ['Document', 'Harness', 'Policy', 'PublicRequest', 'SkillLibrary',
           'make_request', 'recommended_policy', 'ContextAgent', 'MemoryItem',
           'MemoryStore', 'RefinementLoop', 'Verification', 'prepare_dataset_request']
