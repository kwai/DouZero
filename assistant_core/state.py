from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from .cards import FULL_DECK, parse_cards, validate_physical


class Seat(str, Enum):
    LANDLORD = "landlord"
    LANDLORD_UP = "landlord_up"
    LANDLORD_DOWN = "landlord_down"


@dataclass(frozen=True)
class Play:
    seat: Seat
    cards: tuple[str, ...]
    frame_id: int


@dataclass
class RoundState:
    me: Seat
    hand: Counter
    bottom: Counter = field(default_factory=Counter)
    played: Counter = field(default_factory=Counter)
    history: list[Play] = field(default_factory=list)
    uncertain_reason: str | None = None

    @classmethod
    def start(cls, me: Seat, hand, bottom=()):
        hand_c = validate_physical(hand)
        bottom_c = validate_physical(bottom)
        if sum(bottom_c.values()) not in (0, 3):
            raise ValueError("bottom must contain exactly 3 cards when known")
        combined = hand_c + (bottom_c if me == Seat.LANDLORD else Counter())
        expected = 20 if me == Seat.LANDLORD and bottom_c else 17
        if sum(combined.values()) != expected:
            raise ValueError(f"expected {expected} cards, got {sum(combined.values())}")
        validate_physical(combined)
        return cls(me=me, hand=combined, bottom=bottom_c)

    def observe_play(self, seat: Seat, cards, frame_id: int):
        cards_t = parse_cards(cards)
        if not cards_t:
            self.history.append(Play(seat, (), frame_id))
            return
        cards_c = validate_physical(cards_t)
        if seat == self.me:
            if cards_c - self.hand:
                raise ValueError("observed own play is not contained in hand")
            self.hand.subtract(cards_c)
            self.hand += Counter()
        known = self.played + self.hand
        if seat != self.me:
            known += cards_c
        overflow = known - FULL_DECK
        if overflow:
            self.uncertain_reason = f"54-card invariant violated: {dict(overflow)}"
            raise ValueError(self.uncertain_reason)
        self.played.update(cards_c)
        self.history.append(Play(seat, cards_t, frame_id))

    @property
    def unseen(self) -> Counter:
        result = FULL_DECK - self.hand - self.played
        return result

    @property
    def recommendation_allowed(self) -> bool:
        return self.uncertain_reason is None
