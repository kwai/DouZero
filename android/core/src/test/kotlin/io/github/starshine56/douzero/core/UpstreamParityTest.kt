package io.github.starshine56.douzero.core

import org.json.JSONArray
import org.json.JSONObject
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Compares the Kotlin port with fixtures produced by tools/gen_fixtures.py,
 * i.e. with the upstream Python implementation.
 */
class UpstreamParityTest {
    private val fixtures: JSONObject by lazy {
        JSONObject(javaClass.getResource("/fixtures.json")!!.readText())
    }

    private fun JSONArray.objects() = (0 until length()).map { getJSONObject(it) }
    private fun JSONArray.strings() = (0 until length()).map { getString(it) }

    @Test
    fun moveTypesMatch() {
        for (o in fixtures.getJSONArray("moveTypes").objects()) {
            val mt = moveType(Cards.parse(o.getString("move")))
            val rank = if (o.isNull("rank")) null else o.getInt("rank")
            val len = if (o.isNull("len")) null else o.getInt("len")
            assertEquals(MoveType(o.getInt("type"), rank, len), mt, "move ${o.getString("move")}")
        }
    }

    @Test
    fun legalActionsMatch() {
        for (o in fixtures.getJSONArray("legal").objects()) {
            val got = legalActions(Cards.parse(o.getString("hand")), Cards.parse(o.getString("rival")))
                .map { Cards.format(it) }.toSortedSet()
            val want = o.getJSONArray("actions").strings().toSortedSet()
            assertEquals(want, got, "hand=${o.getString("hand")} rival=${o.getString("rival")}")
        }
    }

    private fun replay(o: JSONObject): GameState {
        val state = GameState(
            Position.fromKey(o.getString("position")),
            Cards.parse(o.getString("initialHand")),
            Cards.parse(o.getString("three")),
        )
        for (m in o.getJSONArray("history").strings()) state.play(Cards.parse(m))
        return state
    }

    @Test
    fun replayedStatesHaveUpstreamLegalActions() {
        for (o in fixtures.getJSONArray("decisions").objects()) {
            val state = replay(o)
            assertEquals(o.getString("hand"), Cards.format(state.hand))
            assertEquals(state.user, state.acting)
            val want = o.getJSONObject("values").keySet().toSortedSet()
            val got = state.legalActions().map { Cards.format(it) }.toSortedSet()
            assertEquals(want, got, "history=${o.getJSONArray("history")}")
        }
    }

    @Test
    fun observationEncodingMatches() {
        var checked = 0
        for (o in fixtures.getJSONArray("decisions").objects()) {
            if (!o.has("obsZ")) continue
            val state = replay(o)
            val actions = o.getJSONArray("obsActions").strings().map { Cards.parse(it) }
            val batch = ObsEncoder.encode(state, actions)
            val z = o.getJSONArray("obsZ")
            assertEquals(z.length(), batch.z.size)
            for (i in 0 until z.length()) assertEquals(z.getInt(i).toFloat(), batch.z[i], "z[$i]")
            val x = o.getJSONArray("obsX")
            for (i in 0 until x.length()) assertEquals(x.getDouble(i).toFloat(), batch.x[i], 1e-6f, "x[$i]")
            checked++
        }
        assertTrue(checked > 0)
    }

    private fun models(): OrtModels {
        val dir = File(System.getProperty("models.dir") ?: "")
        assumeTrue(OrtModels.FILES.all { File(dir, it).exists() }, "ONNX models not exported to $dir")
        return OrtModels({ File(dir, it).readBytes() }, threads = 2)
    }

    @Test
    fun modelValuesAndDecisionsMatch() {
        models().use { ort ->
            val agent = DouZeroAgent(ort.models)
            val mismatches = ArrayList<String>()
            var maxDiff = 0.0
            val decisions = fixtures.getJSONArray("decisions").objects()
            for (o in decisions) {
                val state = replay(o)
                val values = o.getJSONObject("values")
                val legal = state.legalActions()
                for (s in agent.evaluate(state, legal)) {
                    maxDiff = maxOf(maxDiff, abs(values.getDouble(Cards.format(s.action)) - s.value))
                }
                val d = agent.decide(state)
                val want = o.getString("action")
                if (Cards.format(d.action) != want) {
                    val top = d.ranked[0].value
                    val upstreamFinish = o.getDouble("winRate") >= DouZeroAgent.FINISH_NOW * 8
                    val (last, prev) = state.lastTwoMoves()
                    val rival = last.ifEmpty { prev }.ifEmpty { null }
                    val ourPath = FinishSearch.selectOptimal(FinishSearch.search(state.hand, state.otherHandCards(), rival))
                    val explained = when {
                        // upstream takes the runner-up whenever the best value is negative (bomb or not);
                        // the port only does that for bombs
                        top < 0 && want == Cards.format(d.ranked[1].action) -> true
                        // both found a forced finish, the paths just start with different moves
                        upstreamFinish && ourPath != null && !FinishSearch.containsFourTwo(ourPath) -> true
                        else -> false
                    }
                    if (!explained) mismatches += "history=${o.getJSONArray("history")} hand=${o.getString("hand")} want=$want got=${d.text} (${d.note})"
                }
            }
            println("decisions=${decisions.size} maxValueDiff=$maxDiff")
            assertTrue(maxDiff < 1e-3, "value diff $maxDiff")
            if (mismatches.isNotEmpty()) fail(mismatches.joinToString("\n"))
        }
    }

    @Test
    fun bidAndFarmerScoresMatch() {
        models().use { ort ->
            val advisor = BidAdvisor(ort.models, DouZeroAgent(ort.models))
            for (o in fixtures.getJSONArray("hands").objects()) {
                val hand = Cards.parse(o.getString("hand"))
                assertEquals(o.getDouble("bid"), advisor.bidScore(hand), 1e-4)
                assertEquals(o.getDouble("farmer"), advisor.farmerScore(hand), 1e-4)
                assertEquals(o.getDouble("up"), advisor.farmerUpScore(hand), 1e-4)
            }
            val agent = DouZeroAgent(ort.models)
            for (o in fixtures.getJSONArray("landlordScores").objects()) {
                val hand = Cards.parse(o.getString("hand"))
                val three = Cards.parse(o.getString("three"))
                val got = advisor.landlordScore(hand, three)
                val want = o.getDouble("winRate")
                val ranked = agent.decide(GameState(Position.LANDLORD, hand, three)).ranked
                // upstream's "best value < 0 -> runner-up" rule
                val upstreamRunnerUp = ranked[0].value < 0 && abs(ranked[1].value * 8.0 - want) < 1e-2
                assertTrue(abs(got - want) < 1e-2 || upstreamRunnerUp || (got >= 80000 && want >= 80000),
                    "landlord score $want vs $got")
            }
        }
    }
}
