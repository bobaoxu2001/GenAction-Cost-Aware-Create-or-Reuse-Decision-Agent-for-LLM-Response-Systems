"""GenAction: a cost-aware create-or-reuse decision agent for LLM response systems.

A small research prototype inspired by online decision making with generative
action sets (the create-or-reuse setting): at each round an agent either reuses a
stored response (cheap, but risks a mismatch) or pays a fixed cost to create a
new, reusable response.
"""

from .embeddings import Embedder
from .environment import (
    Action,
    CreateOrReuseEnv,
    Decision,
    DecisionKind,
    Query,
    StepResult,
)
from .evaluation import EpisodeResult, run_episode, summarize
from .loss import LossModel, semantic_distance
from .policies import (
    AlwaysCreatePolicy,
    DoublyOptimisticPolicy,
    FixedProbabilityPolicy,
    NearestReusePolicy,
    Policy,
    StaticThresholdPolicy,
)

__version__ = "0.1.0"

__all__ = [
    "Embedder",
    "Action",
    "Query",
    "Decision",
    "DecisionKind",
    "StepResult",
    "CreateOrReuseEnv",
    "LossModel",
    "semantic_distance",
    "EpisodeResult",
    "run_episode",
    "summarize",
    "Policy",
    "AlwaysCreatePolicy",
    "NearestReusePolicy",
    "FixedProbabilityPolicy",
    "StaticThresholdPolicy",
    "DoublyOptimisticPolicy",
]
