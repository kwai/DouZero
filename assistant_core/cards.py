from collections import Counter
from dataclasses import dataclass

RANKS = ("3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "X", "D")
FULL_DECK = Counter({**{r: 4 for r in RANKS[:-2]}, "X": 1, "D": 1})
ALIASES = {"10": "T", "SJ": "X", "BJ": "D", "小王": "X", "大王": "D"}


def normalize(rank: str) -> str:
    value = ALIASES.get(rank.strip().upper(), rank.strip().upper())
    if value not in RANKS:
        raise ValueError(f"unknown rank: {rank}")
    return value


def parse_cards(cards) -> tuple[str, ...]:
    if isinstance(cards, str):
        cards = cards.replace(",", " ").split()
    return tuple(normalize(card) for card in cards)


def validate_physical(cards) -> Counter:
    counts = Counter(parse_cards(cards))
    overflow = counts - FULL_DECK
    if overflow:
        raise ValueError(f"impossible cards: {dict(overflow)}")
    return counts


@dataclass(frozen=True)
class CardDetection:
    rank: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    def normalized(self) -> "CardDetection":
        return CardDetection(normalize(self.rank), self.confidence, self.x, self.y, self.width, self.height)
