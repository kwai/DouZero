package io.github.starshine56.douzero.core

/** One candidate move with its raw model value. */
data class ScoredAction(val action: List<Int>, val value: Float) {
    val text: String get() = if (action.isEmpty()) "Pass" else Cards.format(action)
}

/**
 * @param action  the move to make (empty = pass)
 * @param winRate upstream "win_rate": value * 8 (or 80000 / 160000 for forced finishes)
 * @param ranked  all candidates, best first
 * @param note    which upstream heuristic changed the model's choice, if any
 */
data class Decision(
    val action: List<Int>,
    val winRate: Double,
    val ranked: List<ScoredAction>,
    val note: String? = null,
) {
    val text: String get() = if (action.isEmpty()) "Pass" else Cards.format(action)

    /** "AA = 1.234 | Pass = 0.5 | ..." like the upstream window. */
    fun topText(k: Int = 3): String = ranked.take(k).joinToString(" | ") { "${it.text} = ${String.format(java.util.Locale.ROOT, "%.3f", it.value * 8)}" }
}

/**
 * DouZero agent plus the decision heuristics that the upstream GameEnv.step
 * layers on top of the network (finish the hand when possible, etc.).
 */
class DouZeroAgent(private val models: ModelSet, private val batchSize: Int = 256) {

    /** Model value of every legal action, in [actions] order. */
    fun evaluate(state: GameState, actions: List<List<Int>>): List<ScoredAction> {
        val model = models.value(state.user)
        val out = ArrayList<ScoredAction>(actions.size)
        var start = 0
        while (start < actions.size) {
            val chunk = actions.subList(start, minOf(actions.size, start + batchSize))
            val batch = ObsEncoder.encode(state, chunk)
            val values = model.predict(batch.z, batch.x, batch.n)
            for (i in chunk.indices) out += ScoredAction(chunk[i], values[i])
            start += chunk.size
        }
        return out
    }

    /** Decide the move for [state]; it must be the user's turn. */
    fun decide(state: GameState): Decision {
        val legal = state.legalActions()
        if (legal.isEmpty()) return Decision(emptyList(), 0.0, emptyList(), "无合法出牌")
        val scored = evaluate(state, legal)

        var bestIdx = 0
        for (i in scored.indices) if (scored[i].value > scored[bestIdx].value) bestIdx = i
        var action = scored[bestIdx].action
        var winRate = scored[bestIdx].value.toDouble()
        var note: String? = null

        val handSize = state.hand.size
        if (action.size != handSize) {
            // 对直接出完情况做特判
            for (s in scored) {
                if (s.action.size == handSize) {
                    val t = moveType(s.action).type
                    if (t != MoveTypes.FOUR_TWO_PAIRS && t != MoveTypes.FOUR_TWO) {
                        action = s.action
                        winRate = FINISH_NOW
                        note = "检测到可直接出完出法"
                    }
                }
            }
            if (winRate != FINISH_NOW) {
                val (last, prev) = state.lastTwoMoves()
                val rival = when {
                    last.isNotEmpty() -> last
                    prev.isNotEmpty() -> prev
                    else -> null
                }
                val paths = FinishSearch.search(state.hand, state.otherHandCards(), rival)
                val path = FinishSearch.selectOptimal(paths)
                if (path != null && !FinishSearch.containsFourTwo(path) && action != path[0]) {
                    action = path[0]
                    winRate = FINISH_PATH
                    note = "检测到可直接出完路径"
                }
            }
        }

        val ranked = scored.sortedByDescending { it.value }
        if (ranked.size >= 2) {
            // Upstream: 【炸弹胜率低于0 不允许炸】. Applied only to an actual bomb chosen by the model.
            if (ranked[0].value < 0 && action == ranked[0].action && Cards.isBomb(action) && winRate < FINISH_NOW) {
                action = ranked[1].action
                winRate = ranked[1].value.toDouble()
                note = "炸弹胜率低于0 不允许炸"
            }
            if (action.isEmpty()) {
                if (ranked[1].value * 8 > 1) {
                    action = ranked[1].action
                    winRate = ranked[1].value.toDouble()
                    note = "第二选择胜率大于1直接出"
                }
                if (state.user == Position.LANDLORD && (ranked[0].value - ranked[1].value) * 8 < 0.2 &&
                    ranked[1].value * 8 > 0
                ) {
                    action = ranked[1].action
                    winRate = ranked[1].value.toDouble()
                    note = "地主第二选择胜率大于0，且与第一选择相差小于0.2直接出"
                }
            }
        }
        return Decision(action, winRate * 8, ranked, note)
    }

    companion object {
        const val FINISH_NOW = 10000.0
        const val FINISH_PATH = 20000.0
    }
}
