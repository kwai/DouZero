package io.github.starshine56.douzero.core

/**
 * Port of search_utility.py: look for a sequence of moves that no other player
 * can beat and that empties the hand ("可直接出完路径").
 */
object FinishSearch {
    private const val MAX_PATHS = 100
    private val TYPE_RANGE = intArrayOf(4, 5, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

    private class Ctx(val otherCards: List<Int>, val nodeBudget: Int) {
        val paths = ArrayList<List<List<Int>>>()
        var nodes = 0
        val otherGen = MovesGener(otherCards)
        val otherBombs = otherGen.bombs() + otherGen.kingBomb()
        val otherByType = HashMap<Int, List<List<Int>>>()
        fun otherMoves(type: Int) = otherByType.getOrPut(type) {
            if (type == MoveTypes.BOMB) otherBombs else otherGen.genMovesByType(type)
        }
    }

    fun search(myCards: List<Int>, otherCards: List<Int>, rivalMove: List<Int>?, nodeBudget: Int = 200_000): List<List<List<Int>>> {
        val ctx = Ctx(otherCards.sorted(), nodeBudget)
        recurse(myCards.sorted(), ctx, rivalMove, null)
        return ctx.paths
    }

    private fun violatesRival(move: List<Int>, rivalInfo: MoveType): Boolean {
        val info = moveType(move)
        if (info.rank != null && rivalInfo.rank != null && info.rank <= rivalInfo.rank) return true
        if (info.len != null && info.len != rivalInfo.len) return true
        if (rivalInfo.type == MoveTypes.KING_BOMB) return true
        return false
    }

    private fun recurse(myCards: List<Int>, ctx: Ctx, rivalIn: List<Int>?, prev: List<List<Int>>?) {
        if (ctx.paths.size > MAX_PATHS) return
        if (++ctx.nodes > ctx.nodeBudget) return
        val myGen = MovesGener(myCards)
        var rival = rivalIn
        var rivalInfo: MoveType? = null
        var types = TYPE_RANGE
        if (rival != null) {
            if (rival.isNotEmpty()) {
                rivalInfo = moveType(rival)
                types = intArrayOf(4, 5, rivalInfo.type)
            } else {
                rival = null
            }
        }
        for (type in types) {
            val myMoves = myGen.genMovesByType(type)
            if (myMoves.isEmpty()) continue
            val otherMoves = ctx.otherMoves(type)
            for (move in myMoves) {
                if (move.size != myCards.size) {
                    if (type != MoveTypes.BOMB && type != MoveTypes.KING_BOMB && ctx.otherBombs.isNotEmpty()) break
                    if (MoveSelector.filter(type, otherMoves, move).isEmpty()) {
                        if (rival != null && rivalInfo != null && type != MoveTypes.KING_BOMB) {
                            if (violatesRival(move, rivalInfo)) continue
                        }
                        val newCards = Cards.subtract(myCards, move)
                        val newPrev = (prev ?: emptyList()) + listOf(move)
                        recurse(newCards, ctx, null, newPrev)
                    }
                } else {
                    if (rival != null && rivalInfo != null && violatesRival(move, rivalInfo)) continue
                    ctx.paths.add((prev ?: emptyList()) + listOf(move))
                }
            }
        }
    }

    fun evalPath(path: List<List<Int>>): Double {
        var bombs = 0
        for (a in path) {
            if ((a.contains(Cards.BIG_JOKER) && a.contains(Cards.SMALL_JOKER)) || (a.size == 4 && a.toSet().size == 1)) bombs++
        }
        return 1.0 + bombs - path.size * 0.05
    }

    fun selectOptimal(paths: List<List<List<Int>>>): List<List<Int>>? =
        paths.maxByOrNull { evalPath(it) }?.map { it.sorted() }

    /** True if the path contains a 4+2 / 4+2+2 move (upstream refuses those paths). */
    fun containsFourTwo(path: List<List<Int>>): Boolean = path.any {
        val t = moveType(it).type
        t == MoveTypes.FOUR_TWO || t == MoveTypes.FOUR_TWO_PAIRS
    }
}
