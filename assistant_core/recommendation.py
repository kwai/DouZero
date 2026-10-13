from dataclasses import dataclass
from typing import Protocol
from .state import RoundState


@dataclass(frozen=True)
class Recommendation:
    cards: tuple[str, ...]
    score: float
    reason: str


class Policy(Protocol):
    def rank_actions(self, state: RoundState, top_k: int = 3) -> list[Recommendation]: ...


def recommend_verified(state: RoundState, policy: Policy, top_k: int = 3) -> list[Recommendation]:
    if not state.recommendation_allowed:
        raise RuntimeError(state.uncertain_reason or "round state is uncertain")
    if top_k < 1 or top_k > 10:
        raise ValueError("top_k must be between 1 and 10")
    return policy.rank_actions(state, top_k=top_k)
