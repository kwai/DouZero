import unittest
from assistant_core import CardDetection, MultiFrameStabilizer, RoundState, Seat, parse_cards


class CardsTest(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(parse_cards("10 SJ BJ"), ("T", "X", "D"))

    def test_landlord_hand_and_observations(self):
        state = RoundState.start(
            Seat.LANDLORD,
            "3 3 4 4 5 5 6 6 7 7 8 8 9 9 T J Q",
            "K A 2",
        )
        self.assertEqual(sum(state.hand.values()), 20)
        state.observe_play(Seat.LANDLORD_DOWN, "D", 10)
        self.assertEqual(state.played["D"], 1)
        self.assertTrue(state.recommendation_allowed)

    def test_rejects_impossible_observation(self):
        state = RoundState.start(
            Seat.LANDLORD_UP,
            "3 3 3 3 4 4 5 5 6 6 7 7 8 8 9 T J",
        )
        with self.assertRaises(ValueError):
            state.observe_play(Seat.LANDLORD, "3", 11)
        self.assertFalse(state.recommendation_allowed)

    def test_own_play_removes_hand(self):
        state = RoundState.start(
            Seat.LANDLORD_DOWN,
            "3 3 4 4 5 5 6 6 7 7 8 8 9 9 T J Q",
        )
        state.observe_play(Seat.LANDLORD_DOWN, "3 3", 1)
        self.assertEqual(state.hand["3"], 0)


class VisionTest(unittest.TestCase):
    def test_requires_three_frame_vote(self):
        s = MultiFrameStabilizer(window=3)
        frame = [CardDetection("10", .95, 10, 0, 5, 8), CardDetection("A", .94, 20, 0, 5, 8)]
        self.assertFalse(s.push(frame).stable)
        self.assertFalse(s.push(frame).stable)
        out = s.push(frame)
        self.assertTrue(out.stable)
        self.assertEqual(out.ranks, ("T", "A"))

    def test_filters_low_confidence_animation(self):
        s = MultiFrameStabilizer(window=1)
        out = s.push([CardDetection("3", .4, 0, 0, 1, 1)])
        self.assertFalse(out.stable)


if __name__ == "__main__":
    unittest.main()
