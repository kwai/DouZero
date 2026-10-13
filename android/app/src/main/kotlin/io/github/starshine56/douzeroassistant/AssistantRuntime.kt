package io.github.starshine56.douzeroassistant

import android.content.Context
import android.graphics.Bitmap
import io.github.starshine56.douzero.vision.Layout
import io.github.starshine56.douzero.vision.Regions
import io.github.starshine56.douzero.vision.TemplateStore
import io.github.starshine56.douzero.vision.Vision
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicReference

/** In-memory recognition runtime. No frame, card state, or identifier is uploaded or persisted. */
object AssistantRuntime {
    private lateinit var app: Context
    private var vision: Vision? = null
    private val state = AtomicReference(AssistantSnapshot.initial())
    private val votes = HashMap<String, RowVote>()

    @JvmStatic fun initialize(context: Context, openCvLoaded: Boolean) {
        app = context.applicationContext
        if (!openCvLoaded) {
            publish(AssistantSnapshot("OpenCV 初始化失败", "", "", "", "", "--", "--", "识牌不可用"))
            return
        }
        val templates = TemplateStore { name ->
            runCatching { app.assets.open("templates/$name.png").use { it.readBytes() } }.getOrNull()
        }
        vision = Vision(templates, Layout())
        for (name in listOf("hand", "bottom", "left", "right")) votes[name] = RowVote(3)
        publish(AssistantSnapshot("等待横屏牌桌", "", "", "", "", "模型未安装", "--", "等待稳定识别"))
    }

    @JvmStatic fun process(bitmap: Bitmap) {
        val detector = vision ?: return
        val rgba = Mat()
        val bgr = Mat()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
            detector.setDeviceImage(bgr)
            val hand = votes.getValue("hand").push(detector.cards('m', Regions.MY_HAND).sortedText())
            val bottom = votes.getValue("bottom").push(detector.landlordCards().sortedText())
            val left = votes.getValue("left").push(detector.cards('o', Regions.LEFT_PLAYED).sortedText())
            val right = votes.getValue("right").push(detector.cards('o', Regions.RIGHT_PLAYED).sortedText())
            val stable = hand.stable && hand.value.length in 17..20
            val status = when {
                !stable -> "识别中：等待手牌连续稳定"
                left.stable && right.stable -> "牌面已稳定；等待模型和牌局接线"
                else -> "手牌已稳定；等待桌面出牌稳定"
            }
            val remaining = if (stable) remainingText(hand.value, bottom.value, left.value, right.value) else "等待稳定识别"
            publish(AssistantSnapshot(status, hand.value, bottom.value, left.value, right.value,
                "模型未安装", "未生成候选", remaining))
        } catch (t: Throwable) {
            publish(state.get().let { AssistantSnapshot("识牌错误：${t.javaClass.simpleName}", it.hand(), it.bottom(), it.leftPlay(), it.rightPlay(), "--", "--", "已暂停推荐") })
        } finally {
            rgba.release(); bgr.release(); bitmap.recycle()
        }
    }

    private fun remainingText(vararg rows: String): String {
        val order = "DX2AKQJT9876543"
        val full = linkedMapOf<Char, Int>().apply { order.forEach { put(it, if (it == 'D' || it == 'X') 1 else 4) } }
        rows.forEach { row -> row.forEach { c -> if (c in full) full[c] = (full.getValue(c) - 1).coerceAtLeast(0) } }
        val labels = order.map { card -> if (card == 'T') "10" else card.toString() }.joinToString(" ")
        val counts = order.toCharArray().map { card -> full.getValue(card).toString() }.joinToString("  ")
        return "$labels\n$counts"
    }

    @JvmStatic fun current(): AssistantSnapshot = state.get()
    private fun publish(snapshot: AssistantSnapshot) { state.set(snapshot) }

    private class RowVote(private val size: Int) {
        private val values = ArrayDeque<String>()
        data class Result(val value: String, val stable: Boolean)
        fun push(value: String): Result {
            values.addLast(value); while (values.size > size) values.removeFirst()
            val winner = values.groupingBy { it }.eachCount().maxByOrNull { it.value }
            return Result(winner?.key.orEmpty(), values.size == size && (winner?.value ?: 0) >= size / 2 + 1)
        }
    }
}
