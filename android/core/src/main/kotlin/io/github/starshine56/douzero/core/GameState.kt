package io.github.starshine56.douzero.core

import java.util.EnumMap

/**
 * The game as seen by one player ([user]): a port of the parts of the upstream
 * GameEnv that the bot uses. Opponents' hands are unknown; only their card
 * counts and played cards are tracked.
 *
 * Moves must be fed in play order with [play]; [acting] tells whose move is expected.
 */
class GameState(
    val user: Position,
    userHand: List<Int>,
    threeLandlordCards: List<Int>,
) {
    val hand: MutableList<Int> = userHand.sorted().toMutableList()
    val threeLandlordCards: List<Int> = threeLandlordCards.sorted()

    init {
        require(hand.size <= 20) { "hand cannot exceed 20 cards, got ${hand.size}" }
        require(this.threeLandlordCards.size in setOf(0, 3)) { "bottom cards must be unknown or contain exactly 3 cards" }
        val known = Cards.counts(hand)
        for (i in known.indices) {
            val limit = if (Cards.RANKS[i] in setOf(Cards.SMALL_JOKER, Cards.BIG_JOKER)) 1 else 4
            require(known[i] <= limit) { "initial hand exceeds physical deck" }
        }
    }

    /** Landlord cards that the landlord has not played yet (DouZero feature). */
    val threeRemaining: MutableList<Int> = threeLandlordCards.sorted().toMutableList()

    val played: EnumMap<Position, MutableList<Int>> =
        EnumMap<Position, MutableList<Int>>(Position::class.java).apply { Position.entries.forEach { put(it, ArrayList()) } }

    val cardsLeft: EnumMap<Position, Int> = EnumMap<Position, Int>(Position::class.java).apply {
        put(Position.LANDLORD, 20)
        put(Position.LANDLORD_UP, 17)
        put(Position.LANDLORD_DOWN, 17)
        put(user, hand.size)
    }

    /** Every move so far, in order (pass = empty list). */
    val history: MutableList<Pair<Position, List<Int>>> = ArrayList()

    var acting: Position = Position.LANDLORD
        private set

    var bombNum: Int = 0
        private set

    val isOver: Boolean get() = cardsLeft.values.any { it <= 0 }

    /** Cards [user] played that were not in the tracked hand (recognition problems). */
    var mismatches: Int = 0
        private set

    /** The move the acting player has to beat (empty = free lead). */
    fun rivalMove(): List<Int> {
        if (history.isEmpty()) return emptyList()
        val last = history[history.size - 1].second
        if (last.isNotEmpty()) return last
        if (history.size < 2) return emptyList()
        return history[history.size - 2].second
    }

    /** Port of get_last_two_moves: [most recent, previous]. */
    fun lastTwoMoves(): List<List<Int>> {
        val a = if (history.isNotEmpty()) history[history.size - 1].second else emptyList()
        val b = if (history.size >= 2) history[history.size - 2].second else emptyList()
        return listOf(a, b)
    }

    /** Full deck minus everything played minus our hand. */
    fun otherHandCards(): List<Int> {
        val seen = ArrayList<Int>(hand)
        for (p in played.values) seen.addAll(p)
        return Cards.subtract(Cards.FULL_DECK, seen).sorted()
    }

    fun legalActions(): List<List<Int>> = legalActions(hand, rivalMove())

    /** Record the acting player's move and advance the turn. */
    fun play(actionIn: List<Int>) {
        val action = actionIn.sorted()
        val pos = acting
        if (Cards.isBomb(action)) bombNum++
        history.add(pos to action)
        if (pos == user) {
            val check = hand.toMutableList()
            require(action.all { check.remove(it) }) { "own action contains cards not present in hand" }
            hand.clear()
            hand.addAll(check)
            cardsLeft[pos] = hand.size
        } else {
            cardsLeft[pos] = (cardsLeft.getValue(pos) - action.size).coerceAtLeast(0)
        }
        played.getValue(pos).addAll(action)
        if (pos == Position.LANDLORD) {
            for (c in action) {
                if (threeRemaining.isEmpty()) break
                threeRemaining.remove(c)
            }
        }
        acting = pos.next()
    }

    fun copy(): GameState {
        val s = GameState(user, hand, threeLandlordCards)
        s.threeRemaining.clear()
        s.threeRemaining.addAll(threeRemaining)
        for (p in Position.entries) {
            s.played.getValue(p).addAll(played.getValue(p))
            s.cardsLeft[p] = cardsLeft.getValue(p)
        }
        s.history.addAll(history)
        s.acting = acting
        s.bombNum = bombNum
        s.mismatches = mismatches
        return s
    }
}
