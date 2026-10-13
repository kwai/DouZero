package io.github.starshine56.douzero.core

/**
 * Card encoding shared with DouZero:
 * env cards are ints 3..14 (3..A), 17 (2), 20 (small joker), 30 (big joker);
 * "real" cards are the characters 3456789TJQKA2XD used by the upstream project.
 */
object Cards {
    const val SMALL_JOKER = 20
    const val BIG_JOKER = 30

    /** All 15 ranks in ascending order. */
    val RANKS = intArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30)

    /** Real-card characters, index-aligned with [RANKS]. */
    const val REAL = "3456789TJQKA2XD"

    /** Upstream display / recognition order (big to small). */
    const val REAL_DESC = "DX2AKQJT9876543"

    /** The full 54-card deck in env encoding. */
    val FULL_DECK: List<Int> = buildList {
        for (r in 3..14) repeat(4) { add(r) }
        repeat(4) { add(17) }
        add(SMALL_JOKER)
        add(BIG_JOKER)
    }

    fun rankIndex(card: Int): Int {
        val i = RANKS.indexOf(card)
        require(i >= 0) { "unknown env card $card" }
        return i
    }

    fun toEnv(c: Char): Int {
        val i = REAL.indexOf(c.uppercaseChar())
        require(i >= 0) { "unknown card '$c'" }
        return RANKS[i]
    }

    fun toReal(card: Int): Char = REAL[rankIndex(card)]

    fun isCardChar(c: Char): Boolean = REAL.indexOf(c.uppercaseChar()) >= 0

    /** "33TJD" -> sorted env list, rejecting physically impossible multiplicities. */
    fun parse(s: String): List<Int> {
        val cards = s.map(::toEnv).sorted()
        val counts = counts(cards)
        for (i in counts.indices) {
            val limit = if (RANKS[i] == SMALL_JOKER || RANKS[i] == BIG_JOKER) 1 else 4
            require(counts[i] <= limit) { "too many ${toReal(RANKS[i])} cards" }
        }
        return cards
    }

    /** Env list -> ascending real string, e.g. "3345TT". Empty string for pass. */
    fun format(cards: Collection<Int>): String =
        cards.sorted().joinToString("") { toReal(it).toString() }

    /** Env list -> descending real string, e.g. "DX2AA" (the way the game shows a hand). */
    fun formatDesc(cards: Collection<Int>): String =
        cards.sortedDescending().joinToString("") { toReal(it).toString() }

    /** Real string -> descending real string. */
    fun sortDesc(s: String): String = formatDesc(parse(s))

    /** Multiset difference: [from] minus [remove] (each element removed at most once per occurrence). */
    fun subtract(from: Collection<Int>, remove: Collection<Int>): MutableList<Int> {
        val left = from.toMutableList()
        for (c in remove) left.remove(c)
        return left
    }

    /** Count of each rank (index-aligned with [RANKS]). */
    fun counts(cards: Collection<Int>): IntArray {
        val out = IntArray(RANKS.size)
        for (c in cards) out[rankIndex(c)]++
        return out
    }

    fun isBomb(move: List<Int>): Boolean =
        (move.size == 4 && move.all { it == move[0] }) ||
            (move.size == 2 && move.contains(SMALL_JOKER) && move.contains(BIG_JOKER))
}

/** Seat roles. Play order is landlord -> landlord_down -> landlord_up -> landlord. */
enum class Position(val key: String, val label: String) {
    LANDLORD("landlord", "地主"),
    LANDLORD_UP("landlord_up", "地主上家"),
    LANDLORD_DOWN("landlord_down", "地主下家");

    /** The player who acts after this one. */
    fun next(): Position = when (this) {
        LANDLORD -> LANDLORD_DOWN
        LANDLORD_DOWN -> LANDLORD_UP
        LANDLORD_UP -> LANDLORD
    }

    companion object {
        fun fromKey(key: String): Position = entries.first { it.key == key }
    }
}
