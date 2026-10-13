package io.github.starshine56.douzero.core

/**
 * Port of douzero/env/env.py `_get_obs_resnet`: builds the inputs of the
 * ResNet card-play model for every candidate action.
 *
 * z: [n, 40, 54]  rows = action, cards-left one-hots, my hand, other hands,
 *                 unplayed landlord cards, played cards x3, last 32 moves
 * x: [n, 15]      bid_info (12) + multiply_info (3), constants per position
 */
object ObsEncoder {
    const val Z_ROWS = 40
    const val Z_COLS = 54
    const val X_LEN = 15
    private const val SEQ_LEN = 32

    // InfoSet defaults from douzero/env/game.py (the env never overwrites them).
    private val LANDLORD_X = floatArrayOf(
        1f, 0.5f, 1f, 1f, 1f, 1f, 1f, 5f, -4f, 1f, 1f, 1f,
        1f, 0.8f, 1.3f,
    )
    private val FARMER_X = floatArrayOf(
        1f, 0.2f, 1f, 1f, 3.5f, 1f, 1f, 5f, 4f, 1.035f, 1f, 0.15f,
        1f, 2.5f, 1.3f,
    )

    fun xFor(position: Position): FloatArray =
        if (position == Position.LANDLORD) LANDLORD_X.copyOf() else FARMER_X.copyOf()

    /** Port of `_cards2array`: 4x13 count matrix in column-major order + 2 joker flags. */
    fun cardsToArray(cards: List<Int>, out: FloatArray, offset: Int) {
        java.util.Arrays.fill(out, offset, offset + Z_COLS, 0f)
        if (cards.isEmpty()) return
        val counts = Cards.counts(cards)
        for (col in 0 until 13) {
            val n = counts[col].coerceAtMost(4)
            for (row in 0 until n) out[offset + col * 4 + row] = 1f
        }
        if (counts[13] > 0) out[offset + 52] = 1f
        if (counts[14] > 0) out[offset + 53] = 1f
    }

    /** Port of `_get_one_hot_array` (num = 0 wraps to the last slot like Python's index -1). */
    private fun oneHot(num: Int, size: Int, out: FloatArray, offset: Int) {
        val idx = Math.floorMod(num - 1, size).coerceIn(0, size - 1)
        out[offset + idx] = 1f
    }

    /** The 39 action-independent rows (rows 1..39 of z). */
    fun sharedRows(state: GameState): FloatArray {
        val rows = FloatArray((Z_ROWS - 1) * Z_COLS)
        // row 0: cards left (20 + 17 + 17)
        oneHot(state.cardsLeft.getValue(Position.LANDLORD), 20, rows, 0)
        oneHot(state.cardsLeft.getValue(Position.LANDLORD_UP), 17, rows, 20)
        oneHot(state.cardsLeft.getValue(Position.LANDLORD_DOWN), 17, rows, 37)
        cardsToArray(state.hand, rows, 1 * Z_COLS)
        cardsToArray(state.otherHandCards(), rows, 2 * Z_COLS)
        cardsToArray(state.threeRemaining, rows, 3 * Z_COLS)
        cardsToArray(state.played.getValue(Position.LANDLORD), rows, 4 * Z_COLS)
        cardsToArray(state.played.getValue(Position.LANDLORD_UP), rows, 5 * Z_COLS)
        cardsToArray(state.played.getValue(Position.LANDLORD_DOWN), rows, 6 * Z_COLS)

        // rows 7..38: last 32 moves, most recent first, -1 padding in front
        val moves = state.history.map { it.second }.takeLast(SEQ_LEN).reversed()
        val pad = SEQ_LEN - moves.size
        val base = 7 * Z_COLS
        java.util.Arrays.fill(rows, base, base + pad * Z_COLS, -1f)
        for ((i, m) in moves.withIndex()) cardsToArray(m, rows, base + (pad + i) * Z_COLS)
        return rows
    }

    class Batch(val z: FloatArray, val x: FloatArray, val n: Int)

    fun encode(state: GameState, actions: List<List<Int>>, position: Position = state.user): Batch {
        val n = actions.size
        val shared = sharedRows(state)
        val zSize = Z_ROWS * Z_COLS
        val z = FloatArray(n * zSize)
        val x = FloatArray(n * X_LEN)
        val xRow = xFor(position)
        for ((i, a) in actions.withIndex()) {
            cardsToArray(a, z, i * zSize)
            System.arraycopy(shared, 0, z, i * zSize + Z_COLS, shared.size)
            System.arraycopy(xRow, 0, x, i * X_LEN, X_LEN)
        }
        return Batch(z, x, n)
    }
}

/** Port of RealToOnehot used by the bid / farmer networks: 4x15 thresholds, row-major. */
object HandEncoder {
    fun onehot60(cards: List<Int>): FloatArray {
        val counts = Cards.counts(cards)
        val out = FloatArray(60)
        for (i in 0 until 15) for (r in 0 until counts[i].coerceAtMost(4)) out[r * 15 + i] = 1f
        return out
    }
}
