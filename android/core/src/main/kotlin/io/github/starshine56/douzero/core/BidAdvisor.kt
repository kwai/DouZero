package io.github.starshine56.douzero.core

/** Thresholds from the upstream data.json (same names and defaults). */
data class Thresholds(
    val bid1: Double = 0.2, // 叫地主
    val bid2: Double = 0.4, // 抢地主（自己叫过）
    val bid3: Double = 0.5, // 抢地主（自己没叫过）
    val jiabei1: Double = 0.5, // 地主（未被抢）超级加倍
    val jiabei2: Double = 0.4, // 地主（未被抢）加倍
    val jiabei3: Double = 0.6, // 地主（抢来的）超级加倍
    val jiabei4: Double = 0.5, // 地主（抢来的）加倍
    val jiabei5: Double = 1.3, // 农民 超级加倍（upstream 未使用）
    val jiabei6: Double = 1.1, // 农民 加倍（upstream 未使用）
    val jiabei7: Double = 1.2, // 农民 超级加倍
    val jiabei8: Double = 1.0, // 农民 加倍
    val mingpai: Double = 1.5, // 地主明牌
)

enum class DoubleChoice(val label: String) { SUPER("超级加倍"), DOUBLE("加倍"), NONE("不加倍") }

/** Bidding / doubling decisions, ported from choose_multiples_stage in upstream main.py. */
class BidAdvisor(private val models: ModelSet, private val agent: DouZeroAgent) {

    private fun hand(model: HandModel, cards: List<Int>): Double =
        model.predict(HandEncoder.onehot60(cards), 1)[0].toDouble()

    /** BidModel.predict_score: 叫牌得分. */
    fun bidScore(cards: List<Int>): Double = hand(models.bid, cards)

    /** FarmerModel.predict(cards, "farmer"): 不叫得分 (display only upstream). */
    fun farmerScore(cards: List<Int>): Double = hand(models.farmer, cards)

    /** FarmerModel.predict(cards, "up"): pre-game farmer score used for doubling. */
    fun farmerUpScore(cards: List<Int>): Double = hand(models.farmerUp, cards)

    /** LandlordModel.predict_by_model: DouZero's value of the landlord's first move. */
    fun landlordScore(hand20: List<Int>, three: List<Int>): Double =
        agent.decide(GameState(Position.LANDLORD, hand20, three)).winRate

    fun shouldCall(score: Double, t: Thresholds) = score > t.bid1

    fun shouldRob(score: Double, alreadyBid: Boolean, t: Thresholds) =
        score > (if (alreadyBid) t.bid2 else t.bid3)

    /**
     * @param stolen true if our last bidding decision was at a 抢地主 prompt (upstream is_stolen)
     * @param superAvailable whether the 超级加倍 button is on screen
     */
    fun doubleChoice(
        isLandlord: Boolean,
        winRate: Double,
        stolen: Boolean,
        cards: List<Int>,
        superAvailable: Boolean,
        t: Thresholds,
    ): DoubleChoice {
        val (high, low) = if (isLandlord) {
            if (stolen) t.jiabei3 to t.jiabei4 else t.jiabei1 to t.jiabei2
        } else {
            t.jiabei7 to t.jiabei8
        }
        return when {
            winRate > high -> if (superAvailable && hasStrongCards(cards)) DoubleChoice.SUPER else DoubleChoice.DOUBLE
            winRate > low -> DoubleChoice.DOUBLE
            else -> DoubleChoice.NONE
        }
    }

    fun shouldShowCards(isLandlord: Boolean, winRate: Double, t: Thresholds) = isLandlord && winRate > t.mingpai

    companion object {
        /** Upstream checks these substrings of the descending hand string before 超级加倍. */
        fun hasStrongCards(cards: List<Int>): Boolean {
            val s = Cards.formatDesc(cards)
            return listOf("DX", "D2", "X22", "222A", "22AAA").any { it in s }
        }

        /** Upstream "明牌开始" condition: both jokers and three 2s. */
        fun startWithShownCards(cards: List<Int>): Boolean = "DX222" in Cards.formatDesc(cards)
    }
}
