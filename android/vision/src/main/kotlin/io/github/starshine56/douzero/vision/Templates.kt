package io.github.starshine56.douzero.vision

import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ConcurrentHashMap

/**
 * Template images (BGR) at the work scale (screen height 810), looked up by
 * the upstream file name without ".png" (e.g. "m3", "jiaodizhu_btn").
 *
 * [load] returns PNG bytes: the app checks user-captured templates first and
 * falls back to the bundled ones (phone version, see tools/make_mobile_templates.py).
 */
class TemplateStore(private val load: (String) -> ByteArray?) {
    private val cache = ConcurrentHashMap<String, Mat>()
    private val grayCache = ConcurrentHashMap<String, Mat>()
    private val missing = ConcurrentHashMap.newKeySet<String>()

    operator fun get(name: String): Mat? {
        cache[name]?.let { return it }
        if (name in missing) return null
        val bytes = load(name)
        val mat = bytes?.let { Imgcodecs.imdecode(MatOfByte(*it), Imgcodecs.IMREAD_COLOR) }
        if (mat == null || mat.empty()) {
            missing += name
            return null
        }
        cache[name] = mat
        return mat
    }

    /** Grey version of [get], for the card glyphs. */
    fun gray(name: String): Mat? {
        grayCache[name]?.let { return it }
        val bgr = get(name) ?: return null
        val g = Mat()
        Imgproc.cvtColor(bgr, g, Imgproc.COLOR_BGR2GRAY)
        grayCache[name] = g
        return g
    }

    /**
     * Drop cached images after templates were (re)captured. The old Mats are
     * left to the GC because the bot thread may still be matching with them.
     */
    fun invalidate(name: String? = null) {
        if (name == null) {
            cache.clear()
            grayCache.clear()
            missing.clear()
        } else {
            cache.remove(name)
            grayCache.remove(name)
            missing.remove(name)
        }
    }

    companion object {
        /** Card template sets: m = my hand, o = opponents' played, c = my played, z = landlord cards. */
        val CARD_SETS = mapOf('m' to "我的手牌", 'o' to "对手出的牌", 'c' to "我出的牌", 'z' to "底牌")

        val BUTTONS = linkedMapOf(
            "jiaodizhu_btn" to "叫地主", "bujiao_btn" to "不叫", "qiangdizhu_btn" to "抢地主", "buqiang_btn" to "不抢",
            "jiabei_btn" to "加倍", "bujiabei_btn" to "不加倍", "chaojijiabei_btn" to "超级加倍",
            "mingpai_btn" to "明牌（加倍后）", "ming_btn" to "明牌开始",
            "play_card" to "出牌", "pass_btn" to "不出（按钮）", "yaobuqi" to "要不起", "buchu" to "“不出”提示文字",
            "chat" to "聊天按钮", "laotou" to "底牌背面（开局）", "over" to "结算豆子",
            "continue" to "继续游戏", "win" to "胜利", "lose" to "失败", "quick_start" to "快速开始",
            "reward" to "领取奖励", "start_game" to "开始游戏", "tuoguan" to "取消托管", "sure" to "确定",
            "tuijian" to "推荐", "good" to "好的", "zhidao" to "知道了", "chacha" to "关闭×", "fangqi" to "放弃（返豆暴击弹窗）",
            "fangqi_win" to "放弃（赢豆暴击弹窗）", "fangqi_star" to "放弃（星星保护弹窗）",
            "clock" to "闹钟（上下家）", "clock_me" to "闹钟（自己）",
        )

        fun cardTemplateNames(): List<String> =
            CARD_SETS.keys.flatMap { set -> "3456789TJQKA2XD".map { "$set$it" } }

        fun allNames(): List<String> = cardTemplateNames() + BUTTONS.keys
    }
}
