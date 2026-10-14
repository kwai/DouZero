package io.github.starshine56.douzero.vision

import org.opencv.core.Rect

/**
 * How a region follows the screen when the device is wider than 16:9.
 * The game UI scales with the screen height; centre-anchored widgets stay
 * around the middle while edge-anchored ones stick to the left/right edge.
 */
enum class Anchor(val label: String) { CENTER("居中"), LEFT("靠左"), RIGHT("靠右") }

/**
 * A region in the reference space (1440x810, the upstream window size) plus
 * its anchor; [Layout.toWork] places it on the actual, usually wider, screen.
 */
data class RegionSpec(val x: Int, val y: Int, val w: Int, val h: Int, val anchor: Anchor = Anchor.CENTER) {
    fun serialize(): String = "$x,$y,$w,$h,${anchor.name}"

    companion object {
        fun parse(s: String): RegionSpec? = runCatching {
            val p = s.split(',')
            RegionSpec(p[0].trim().toInt(), p[1].trim().toInt(), p[2].trim().toInt(), p[3].trim().toInt(),
                if (p.size > 4) Anchor.valueOf(p[4].trim()) else Anchor.CENTER)
        }.getOrNull()
    }
}

/** Region names and their defaults. */
object Regions {
    const val MY_HAND = "myHand"
    const val MY_PLAYED = "myPlayed"
    const val LEFT_PLAYED = "leftPlayed"
    const val RIGHT_PLAYED = "rightPlayed"
    const val LANDLORD_CARDS = "landlordCards"
    const val MY_PASS = "myPass"
    const val LEFT_PASS = "leftPass"
    const val RIGHT_PASS = "rightPass"
    const val BUTTONS = "buttons"
    const val CHAT = "chat"
    const val BEANS_LEFT = "beansLeft"
    const val BEANS_ME = "beansMe"
    const val BEANS_RIGHT = "beansRight"
    const val CONTINUE = "continue"
    const val RESULT = "result"
    const val QUICK_START = "quickStart"
    const val REWARD = "reward"
    const val START_GAME = "startGame"
    const val TUOGUAN = "tuoguan"
    const val SURE = "sure"
    const val TUIJIAN = "tuijian"
    const val GOOD = "good"
    const val ZHIDAO = "zhidao"
    const val CHACHA = "chacha"
    const val FANGQI = "fangqi"
    const val CLOCK_LEFT = "clockLeft"
    const val CLOCK_RIGHT = "clockRight"
    const val CLOCK_ME = "clockMe"
    const val ANI_LEFT = "aniLeft"
    const val ANI_RIGHT = "aniRight"
    const val ANI_ME = "aniMe"

    /**
     * Defaults for the phone version of the game, measured on a 20:9 screen
     * (work image 1782x810). Card, button and label areas follow the screen
     * edge or the centre like the game UI does. Result screen: 失败 and
     * 继续游戏 from the phone; the other pop-up regions are still the
     * upstream PC ones (centre-anchored).
     */
    val DEFAULTS: LinkedHashMap<String, RegionSpec> = linkedMapOf(
        MY_HAND to RegionSpec(-51, 485, 1540, 130),
        MY_PLAYED to RegionSpec(229, 300, 980, 195),
        // long moves start further out and reach the middle of the table
        LEFT_PLAYED to RegionSpec(240, 175, 651, 130, Anchor.LEFT),
        RIGHT_PLAYED to RegionSpec(549, 175, 689, 130, Anchor.RIGHT),
        // the three cards at the end of the 记牌器 bar, or a bit further right at the top without it
        LANDLORD_CARDS to RegionSpec(785, 5, 165, 80, Anchor.LEFT),
        MY_PASS to RegionSpec(389, 300, 660, 200),
        LEFT_PASS to RegionSpec(270, 200, 290, 120, Anchor.LEFT),
        RIGHT_PASS to RegionSpec(878, 200, 300, 120, Anchor.RIGHT),
        BUTTONS to RegionSpec(209, 360, 1040, 130),
        CHAT to RegionSpec(1238, 740, 165, 60, Anchor.RIGHT),
        BEANS_LEFT to RegionSpec(308, 204, 254, 60),
        BEANS_ME to RegionSpec(295, 474, 264, 60),
        BEANS_RIGHT to RegionSpec(882, 203, 230, 60),
        CONTINUE to RegionSpec(769, 670, 260, 100),
        RESULT to RegionSpec(589, 85, 270, 140),
        FANGQI to RegionSpec(429, 520, 300, 110),
        CLOCK_LEFT to RegionSpec(250, 180, 200, 160, Anchor.LEFT),
        CLOCK_RIGHT to RegionSpec(1008, 180, 200, 160, Anchor.RIGHT),
        CLOCK_ME to RegionSpec(429, 330, 500, 170),
        QUICK_START to RegionSpec(1107, 679, 252, 118),
        REWARD to RegionSpec(587, 570, 224, 86),
        START_GAME to RegionSpec(720, 466, 261, 117),
        TUOGUAN to RegionSpec(577, 613, 271, 128),
        SURE to RegionSpec(630, 456, 289, 143),
        TUIJIAN to RegionSpec(729, 465, 219, 100),
        GOOD to RegionSpec(434, 599, 219, 88),
        ZHIDAO to RegionSpec(593, 543, 224, 94),
        CHACHA to RegionSpec(1036, 65, 300, 230),
        ANI_LEFT to RegionSpec(600, 140, 20, 20, Anchor.LEFT),
        ANI_RIGHT to RegionSpec(838, 140, 20, 20, Anchor.RIGHT),
        ANI_ME to RegionSpec(700, 140, 20, 20),
    )

    val LABELS: Map<String, String> = mapOf(
        MY_HAND to "我的手牌",
        MY_PLAYED to "我的出牌区",
        LEFT_PLAYED to "上家出牌区（左）",
        RIGHT_PLAYED to "下家出牌区（右）",
        LANDLORD_CARDS to "三张底牌",
        MY_PASS to "我的“不出”提示",
        LEFT_PASS to "上家“不出”提示",
        RIGHT_PASS to "下家“不出”提示",
        BUTTONS to "操作按钮区（叫地主/加倍/出牌）",
        CHAT to "聊天按钮（判断在牌桌内）",
        BEANS_LEFT to "结算豆子（左）",
        BEANS_ME to "结算豆子（我）",
        BEANS_RIGHT to "结算豆子（右）",
        CONTINUE to "继续游戏按钮",
        RESULT to "胜利/失败标志",
        QUICK_START to "快速开始按钮",
        REWARD to "领取奖励弹窗",
        START_GAME to "开始游戏按钮",
        TUOGUAN to "取消托管按钮",
        SURE to "确定按钮",
        TUIJIAN to "推荐弹窗",
        GOOD to "好的按钮",
        ZHIDAO to "知道了按钮",
        CHACHA to "弹窗关闭×",
        FANGQI to "弹窗「放弃」（返豆/赢豆暴击、星星保护）",
        CLOCK_LEFT to "闹钟（上家思考中）",
        CLOCK_RIGHT to "闹钟（下家思考中）",
        CLOCK_ME to "闹钟（轮到自己）",
        ANI_LEFT to "动画检测点（左）",
        ANI_RIGHT to "动画检测点（右）",
        ANI_ME to "动画检测点（中）",
    )
}

/**
 * Maps reference regions into the "work" image: the screenshot scaled so its
 * height is [REF_H]. For a 16:9 screen the work image is exactly the upstream
 * 1440x810 window; wider screens get extra width on both sides.
 */
class Layout(overrides: Map<String, RegionSpec> = emptyMap()) {
    val regions: Map<String, RegionSpec> = Regions.DEFAULTS + overrides

    operator fun get(name: String): RegionSpec = regions[name] ?: error("unknown region $name")

    fun with(name: String, spec: RegionSpec): Layout = Layout(regions + (name to spec))

    companion object {
        const val REF_W = 1440
        const val REF_H = 810

        fun toWork(spec: RegionSpec, workWidth: Int): Rect {
            val x = when (spec.anchor) {
                Anchor.CENTER -> workWidth / 2 + (spec.x - REF_W / 2)
                Anchor.LEFT -> spec.x
                Anchor.RIGHT -> workWidth - (REF_W - spec.x)
            }
            return Rect(x, spec.y, spec.w, spec.h)
        }

        fun fromWork(rect: Rect, anchor: Anchor, workWidth: Int): RegionSpec {
            val x = when (anchor) {
                Anchor.CENTER -> rect.x - workWidth / 2 + REF_W / 2
                Anchor.LEFT -> rect.x
                Anchor.RIGHT -> rect.x - workWidth + REF_W
            }
            return RegionSpec(x, rect.y, rect.width, rect.height, anchor)
        }
    }
}
