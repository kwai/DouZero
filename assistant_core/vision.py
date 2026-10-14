from collections import Counter, deque
from dataclasses import dataclass
from .cards import CardDetection


@dataclass(frozen=True)
class StableFrame:
    ranks: tuple[str, ...]
    stable: bool
    confidence: float


class MultiFrameStabilizer:
    """Votes on whole observations; prevents one animation frame corrupting a round."""

    def __init__(self, window: int = 3, min_confidence: float = 0.72):
        if window < 1:
            raise ValueError("window must be positive")
        self.frames = deque(maxlen=window)
        self.window = window
        self.min_confidence = min_confidence

    def push(self, detections: list[CardDetection]) -> StableFrame:
        clean = [d.normalized() for d in detections if d.confidence >= self.min_confidence]
        clean.sort(key=lambda d: d.x)
        ranks = tuple(d.rank for d in clean)
        confidence = min((d.confidence for d in clean), default=0.0)
        self.frames.append(ranks)
        votes = Counter(self.frames)
        winner, count = votes.most_common(1)[0]
        required = self.window // 2 + 1
        stable = len(self.frames) == self.window and count >= required and bool(winner)
        return StableFrame(winner, stable, confidence)

    def reset(self):
        self.frames.clear()
