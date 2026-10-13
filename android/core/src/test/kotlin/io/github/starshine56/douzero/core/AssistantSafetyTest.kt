package io.github.starshine56.douzero.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class AssistantSafetyTest {
    @Test fun fullDeckConservation() {
        val hand = Cards.parse("33445566778899TJQ")
        val state = GameState(Position.LANDLORD_UP, hand, emptyList())
        assertEquals(17, state.hand.size)
        assertEquals(37, state.otherHandCards().size)
    }

    @Test fun legalActionsBelongToHand() {
        val hand = Cards.parse("33445566778899TJQ")
        val state = GameState(Position.LANDLORD_UP, hand, emptyList())
        state.play(emptyList()) // landlord passes in synthetic state
        state.play(emptyList()) // landlord_down passes; now user acts
        assertTrue(state.legalActions().isNotEmpty())
        for (action in state.legalActions()) {
            val remaining = state.hand.toMutableList()
            assertTrue(action.all { remaining.remove(it) })
        }
    }

    @Test fun malformedCardsRejected() {
        assertFailsWith<IllegalArgumentException> { Cards.parse("33333") }
    }
}
