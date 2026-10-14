package io.github.starshine56.douzero.core

/** Move type ids, identical to douzero/env/utils.py. */
object MoveTypes {
    const val PASS = 0
    const val SINGLE = 1
    const val PAIR = 2
    const val TRIPLE = 3
    const val BOMB = 4
    const val KING_BOMB = 5
    const val THREE_ONE = 6
    const val THREE_TWO = 7
    const val SERIAL_SINGLE = 8
    const val SERIAL_PAIR = 9
    const val SERIAL_TRIPLE = 10
    const val SERIAL_3_1 = 11
    const val SERIAL_3_2 = 12
    const val FOUR_TWO = 13
    const val FOUR_TWO_PAIRS = 14
    const val WRONG = 15

    const val MIN_SINGLE_CARDS = 5
    const val MIN_PAIRS = 3
    const val MIN_TRIPLES = 2
}

/** Result of [moveType]; [rank] and [len] are null where upstream omits the key. */
data class MoveType(val type: Int, val rank: Int? = null, val len: Int? = null)

private fun isContinuous(seq: List<Int>): Boolean {
    for (i in 0 until seq.size - 1) if (seq[i + 1] - seq[i] != 1) return false
    return true
}

/** Port of douzero/env/move_detector.py get_move_type. */
fun moveType(moveIn: List<Int>): MoveType {
    val move = moveIn.sorted()
    val size = move.size
    if (size == 0) return MoveType(MoveTypes.PASS)
    if (size == 1) return MoveType(MoveTypes.SINGLE, move[0])

    val counts = LinkedHashMap<Int, Int>()
    for (c in move) counts[c] = (counts[c] ?: 0) + 1

    if (size == 2) {
        if (move[0] == move[1]) return MoveType(MoveTypes.PAIR, move[0])
        if (move[0] == Cards.SMALL_JOKER && move[1] == Cards.BIG_JOKER) return MoveType(MoveTypes.KING_BOMB, 1)
        return MoveType(MoveTypes.WRONG)
    }
    if (size == 3) {
        return if (counts.size == 1) MoveType(MoveTypes.TRIPLE, move[0]) else MoveType(MoveTypes.WRONG)
    }
    if (size == 4) {
        if (counts.size == 1) return MoveType(MoveTypes.BOMB, move[0])
        if (counts.size == 2) {
            if ((move[0] == move[1] && move[1] == move[2]) || (move[1] == move[2] && move[2] == move[3])) {
                return MoveType(MoveTypes.THREE_ONE, move[1])
            }
            return MoveType(MoveTypes.WRONG)
        }
        return MoveType(MoveTypes.WRONG)
    }
    if (isContinuous(move)) return MoveType(MoveTypes.SERIAL_SINGLE, move[0], size)
    if (size == 5) {
        return if (counts.size == 2) MoveType(MoveTypes.THREE_TWO, move[2]) else MoveType(MoveTypes.WRONG)
    }

    // count of counts: how many ranks appear n times
    val countDict = HashMap<Int, Int>()
    for (n in counts.values) countDict[n] = (countDict[n] ?: 0) + 1

    if (size == 6 && (counts.size == 2 || counts.size == 3) && countDict[4] == 1 &&
        (countDict[2] == 1 || countDict[1] == 2)
    ) {
        return MoveType(MoveTypes.FOUR_TWO, move[2])
    }
    if (size == 8 && (((counts.size == 3 || counts.size == 2) && countDict[4] == 1 && countDict[2] == 2) ||
            countDict[4] == 2)
    ) {
        return MoveType(MoveTypes.FOUR_TWO_PAIRS, counts.filter { it.value == 4 }.keys.max())
    }

    val keys = counts.keys.sorted()
    if (counts.size == countDict[2] && isContinuous(keys)) {
        return MoveType(MoveTypes.SERIAL_PAIR, keys[0], keys.size)
    }
    if (counts.size == countDict[3] && isContinuous(keys)) {
        return MoveType(MoveTypes.SERIAL_TRIPLE, keys[0], keys.size)
    }

    if ((countDict[3] ?: 0) >= MoveTypes.MIN_TRIPLES) {
        val serial3 = ArrayList<Int>()
        val single = ArrayList<Int>()
        val pair = ArrayList<Int>()
        for ((k, v) in counts) {
            when (v) {
                3 -> serial3.add(k)
                1 -> single.add(k)
                2 -> pair.add(k)
                else -> return MoveType(MoveTypes.WRONG)
            }
        }
        serial3.sort()
        if (isContinuous(serial3)) {
            if (serial3.size == single.size + pair.size * 2) {
                return MoveType(MoveTypes.SERIAL_3_1, serial3[0], serial3.size)
            }
            if (serial3.size == pair.size && counts.size == serial3.size * 2) {
                return MoveType(MoveTypes.SERIAL_3_2, serial3[0], serial3.size)
            }
        }
        if (serial3.size == 4) {
            if (isContinuous(serial3.subList(1, 4))) return MoveType(MoveTypes.SERIAL_3_1, serial3[1], 3)
            if (isContinuous(serial3.subList(0, 3))) return MoveType(MoveTypes.SERIAL_3_1, serial3[0], 3)
        }
    }
    return MoveType(MoveTypes.WRONG)
}

/** All k-combinations of [items] by position (duplicates in [items] give duplicate combos, like itertools). */
internal fun <T> combinations(items: List<T>, k: Int): List<List<T>> {
    val out = ArrayList<List<T>>()
    if (k < 0 || k > items.size) return out
    val idx = IntArray(k) { it }
    while (true) {
        out.add(List(k) { items[idx[it]] })
        var i = k - 1
        while (i >= 0 && idx[i] == items.size - k + i) i--
        if (i < 0) break
        idx[i]++
        for (j in i + 1 until k) idx[j] = idx[j - 1] + 1
    }
    return out
}

/**
 * Port of douzero/env/move_generator.py MovesGener.
 * Every returned move is sorted ascending and each list is free of duplicates.
 */
class MovesGener(cards: List<Int>) {
    val cards: List<Int> = cards.sorted()
    private val counts = java.util.TreeMap<Int, Int>().apply { for (c in this@MovesGener.cards) merge(c, 1, Int::plus) }

    fun singles(): List<List<Int>> = counts.keys.map { listOf(it) }
    fun pairs(): List<List<Int>> = counts.filter { it.value >= 2 }.keys.map { listOf(it, it) }
    fun triples(): List<List<Int>> = counts.filter { it.value >= 3 }.keys.map { listOf(it, it, it) }
    fun bombs(): List<List<Int>> = counts.filter { it.value == 4 }.keys.map { listOf(it, it, it, it) }
    fun kingBomb(): List<List<Int>> =
        if (counts.containsKey(Cards.SMALL_JOKER) && counts.containsKey(Cards.BIG_JOKER)) {
            listOf(listOf(Cards.SMALL_JOKER, Cards.BIG_JOKER))
        } else emptyList()

    fun threeOne(): List<List<Int>> {
        val out = ArrayList<List<Int>>()
        for (t in singles()) for (tr in triples()) if (t[0] != tr[0]) out.add((t + tr).sorted())
        return out
    }

    fun threeTwo(): List<List<Int>> {
        val out = ArrayList<List<Int>>()
        for (p in pairs()) for (tr in triples()) if (p[0] != tr[0]) out.add((p + tr).sorted())
        return out
    }

    private fun serialMoves(ranks: Collection<Int>, minSerial: Int, repeat: Int, repeatNumIn: Int): List<List<Int>> {
        val repeatNum = if (repeatNumIn < minSerial) 0 else repeatNumIn
        val single = ranks.toSortedSet().toList()
        val runs = ArrayList<Pair<Int, Int>>() // (start, length)
        var start = 0
        var i = 0
        var longest = 1
        while (i < single.size) {
            if (i + 1 < single.size && single[i + 1] - single[i] == 1) {
                longest++
                i++
            } else {
                runs.add(start to longest)
                i++
                start = i
                longest = 1
            }
        }
        val moves = ArrayList<List<Int>>()
        for ((s, len) in runs) {
            if (len < minSerial) continue
            val run = single.subList(s, s + len)
            if (repeatNum == 0) {
                var steps = minSerial
                while (steps <= len) {
                    var index = 0
                    while (steps + index <= len) {
                        moves.add(run.subList(index, index + steps).flatMap { r -> List(repeat) { r } })
                        index++
                    }
                    steps++
                }
            } else {
                if (len < repeatNum) continue
                var index = 0
                while (index + repeatNum <= len) {
                    moves.add(run.subList(index, index + repeatNum).flatMap { r -> List(repeat) { r } })
                    index++
                }
            }
        }
        return moves
    }

    fun serialSingle(repeatNum: Int = 0) = serialMoves(counts.keys, MoveTypes.MIN_SINGLE_CARDS, 1, repeatNum)
    fun serialPair(repeatNum: Int = 0) =
        serialMoves(counts.filter { it.value >= 2 }.keys, MoveTypes.MIN_PAIRS, 2, repeatNum)

    fun serialTriple(repeatNum: Int = 0) =
        serialMoves(counts.filter { it.value >= 3 }.keys, MoveTypes.MIN_TRIPLES, 3, repeatNum)

    fun serial31(repeatNum: Int = 0): List<List<Int>> {
        val out = LinkedHashSet<List<Int>>()
        for (s3 in serialTriple(repeatNum)) {
            val s3set = s3.toSet()
            val rest = cards.filter { it !in s3set }
            for (kick in combinations(rest, s3set.size)) out.add((s3 + kick).sorted())
        }
        return out.toList()
    }

    fun serial32(repeatNum: Int = 0): List<List<Int>> {
        val out = LinkedHashSet<List<Int>>()
        val pairRanks = counts.filter { it.value >= 2 }.keys.sorted()
        for (s3 in serialTriple(repeatNum)) {
            val s3set = s3.toSet()
            val candidates = pairRanks.filter { it !in s3set }
            for (kick in combinations(candidates, s3set.size)) {
                out.add((s3 + kick.flatMap { listOf(it, it) }).sorted())
            }
        }
        return out.toList()
    }

    fun fourTwo(): List<List<Int>> {
        val out = LinkedHashSet<List<Int>>()
        for (fc in counts.filter { it.value == 4 }.keys) {
            val rest = cards.filter { it != fc }
            for (kick in combinations(rest, 2)) out.add((List(4) { fc } + kick).sorted())
        }
        return out.toList()
    }

    fun fourTwoPairs(): List<List<Int>> {
        val out = LinkedHashSet<List<Int>>()
        for (fc in counts.filter { it.value == 4 }.keys) {
            val pairRanks = counts.filter { it.key != fc && it.value >= 2 }.keys.toList()
            for (kick in combinations(pairRanks, 2)) {
                out.add((List(4) { fc } + listOf(kick[0], kick[0], kick[1], kick[1])).sorted())
            }
        }
        return out.toList()
    }

    /** All moves available when leading. */
    fun genMoves(): List<List<Int>> {
        val moves = ArrayList<List<Int>>()
        moves += singles()
        moves += pairs()
        moves += triples()
        moves += bombs()
        moves += kingBomb()
        moves += threeOne()
        moves += threeTwo()
        moves += serialSingle()
        moves += serialPair()
        moves += serialTriple()
        moves += serial31()
        moves += serial32()
        moves += fourTwo()
        moves += fourTwoPairs()
        return moves
    }

    fun genMovesByType(type: Int): List<List<Int>> = when (type) {
        MoveTypes.SINGLE -> singles()
        MoveTypes.PAIR -> pairs()
        MoveTypes.TRIPLE -> triples()
        MoveTypes.BOMB -> bombs()
        MoveTypes.KING_BOMB -> kingBomb()
        MoveTypes.THREE_ONE -> threeOne()
        MoveTypes.THREE_TWO -> threeTwo()
        MoveTypes.SERIAL_SINGLE -> serialSingle()
        MoveTypes.SERIAL_PAIR -> serialPair()
        MoveTypes.SERIAL_TRIPLE -> serialTriple()
        MoveTypes.SERIAL_3_1 -> serial31()
        MoveTypes.SERIAL_3_2 -> serial32()
        MoveTypes.FOUR_TWO -> fourTwo()
        MoveTypes.FOUR_TWO_PAIRS -> fourTwoPairs()
        else -> emptyList()
    }
}

/** Port of douzero/env/move_selector.py: keep the moves (of the rival's type) that beat [rival]. */
object MoveSelector {
    private fun maxRankWithCount(move: List<Int>, n: Int): Int {
        val counts = HashMap<Int, Int>()
        for (c in move) counts[c] = (counts[c] ?: 0) + 1
        return counts.filter { it.value == n }.keys.maxOrNull() ?: 0
    }

    private fun byFirst(moves: List<List<Int>>, rival: List<Int>) = moves.filter { it[0] > rival[0] }

    fun filter(type: Int, moves: List<List<Int>>, rivalIn: List<Int>): List<List<Int>> {
        val rival = rivalIn.sorted()
        return when (type) {
            MoveTypes.SINGLE, MoveTypes.PAIR, MoveTypes.TRIPLE, MoveTypes.BOMB,
            MoveTypes.SERIAL_SINGLE, MoveTypes.SERIAL_PAIR, MoveTypes.SERIAL_TRIPLE -> byFirst(moves, rival)
            MoveTypes.KING_BOMB -> moves
            MoveTypes.THREE_ONE -> moves.filter { it.sorted()[1] > rival[1] }
            MoveTypes.THREE_TWO -> moves.filter { it.sorted()[2] > rival[2] }
            MoveTypes.SERIAL_3_1, MoveTypes.SERIAL_3_2 -> {
                val r = maxRankWithCount(rival, 3)
                moves.filter { maxRankWithCount(it, 3) > r }
            }
            MoveTypes.FOUR_TWO -> moves.filter { it.sorted()[2] > rival[2] }
            MoveTypes.FOUR_TWO_PAIRS -> {
                val r = maxRankWithCount(rival, 4)
                moves.filter { maxRankWithCount(it, 4) > r }
            }
            else -> emptyList()
        }
    }
}

/**
 * Port of GameEnv.get_legal_card_play_actions: legal moves for [hand] when the
 * move to beat is [rivalMove] (empty = free lead). Pass is the empty list.
 */
fun legalActions(hand: List<Int>, rivalMove: List<Int>): List<List<Int>> {
    val mg = MovesGener(hand)
    val rival = rivalMove.sorted()
    val rt = moveType(rival)
    val len = rt.len ?: 1
    var moves: List<List<Int>> = when (rt.type) {
        MoveTypes.PASS -> mg.genMoves()
        MoveTypes.SINGLE -> MoveSelector.filter(rt.type, mg.singles(), rival)
        MoveTypes.PAIR -> MoveSelector.filter(rt.type, mg.pairs(), rival)
        MoveTypes.TRIPLE -> MoveSelector.filter(rt.type, mg.triples(), rival)
        MoveTypes.BOMB -> MoveSelector.filter(rt.type, mg.bombs() + mg.kingBomb(), rival)
        MoveTypes.KING_BOMB -> emptyList()
        MoveTypes.THREE_ONE -> MoveSelector.filter(rt.type, mg.threeOne(), rival)
        MoveTypes.THREE_TWO -> MoveSelector.filter(rt.type, mg.threeTwo(), rival)
        MoveTypes.SERIAL_SINGLE -> MoveSelector.filter(rt.type, mg.serialSingle(len), rival)
        MoveTypes.SERIAL_PAIR -> MoveSelector.filter(rt.type, mg.serialPair(len), rival)
        MoveTypes.SERIAL_TRIPLE -> MoveSelector.filter(rt.type, mg.serialTriple(len), rival)
        MoveTypes.SERIAL_3_1 -> MoveSelector.filter(rt.type, mg.serial31(len), rival)
        MoveTypes.SERIAL_3_2 -> MoveSelector.filter(rt.type, mg.serial32(len), rival)
        MoveTypes.FOUR_TWO -> MoveSelector.filter(rt.type, mg.fourTwo(), rival)
        MoveTypes.FOUR_TWO_PAIRS -> MoveSelector.filter(rt.type, mg.fourTwoPairs(), rival)
        else -> emptyList()
    }
    if (rt.type != MoveTypes.PASS && rt.type != MoveTypes.BOMB && rt.type != MoveTypes.KING_BOMB) {
        moves = moves + mg.bombs() + mg.kingBomb()
    }
    if (rival.isNotEmpty()) moves = moves + listOf(emptyList())
    return moves.map { it.sorted() }.distinct()
}
