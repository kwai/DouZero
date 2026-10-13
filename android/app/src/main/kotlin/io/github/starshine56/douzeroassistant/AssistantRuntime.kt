package io.github.starshine56.douzeroassistant

import android.content.Context
import android.graphics.Bitmap
import io.github.starshine56.douzero.core.Cards
import io.github.starshine56.douzero.core.OrtModels
import io.github.starshine56.douzero.core.Position
import io.github.starshine56.douzero.vision.*
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/** Local-only bridge: MediaProjection -> OpenCV -> observable game -> DouZero -> overlay. */
object AssistantRuntime {
    private lateinit var app: Context
    private val state = AtomicReference(AssistantSnapshot.initial())
    private val frameLock = Any()
    private val latestBgr = Mat()
    private var hasFrame = false
    private val starting = AtomicBoolean(false)
    @Volatile private var engine: BotEngine? = null
    private var models: OrtModels? = null

    @JvmStatic fun initialize(context: Context, openCvLoaded: Boolean) {
        app = context.applicationContext
        publish(if (openCvLoaded) snapshot(status = "等待横屏牌桌")
        else snapshot(status = "OpenCV 初始化失败", remaining = "识牌不可用"))
    }

    /** Called by the latest-frame-only worker; ownership of [bitmap] is transferred here. */
    @JvmStatic fun process(bitmap: Bitmap) {
        val rgba = Mat(); val bgr = Mat()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
            synchronized(frameLock) { bgr.copyTo(latestBgr); hasFrame = true }
            ensureEngine()
        } catch (t: Throwable) {
            publish(snapshot(status = "画面处理错误：${t.javaClass.simpleName}", remaining = "已暂停推荐"))
        } finally {
            rgba.release(); bgr.release(); bitmap.recycle()
        }
    }

    private fun ensureEngine() {
        if (engine != null || !starting.compareAndSet(false, true)) return
        Thread({
            try {
                publish(snapshot(status = "加载本地 DouZero 模型…"))
                val ort = OrtModels({ name -> app.assets.open("models/$name").use { it.readBytes() } }, threads = 4)
                models = ort
                val templates = TemplateStore { name ->
                    runCatching { app.assets.open("templates/$name.png").use { it.readBytes() } }.getOrNull()
                }
                val source = ScreenSource { dst -> synchronized(frameLock) {
                    if (!hasFrame) false else { latestBgr.copyTo(dst); true }
                } }
                val noTap = Tapper { _, _ -> Unit }
                val layout = Layout(mapOf(
                    // Measured from the supplied 2400x1080 independent-app screenshot after scaling to work height 810.
                    Regions.LANDLORD_CARDS to RegionSpec(1040, 5, 110, 45, Anchor.LEFT),
                    Regions.RIGHT_PLAYED to RegionSpec(900, 155, 520, 185, Anchor.LEFT),
                    Regions.MY_HAND to RegionSpec(120, 480, 1450, 190, Anchor.LEFT),
                ))
                val e = BotEngine(source, noTap, Vision(templates, layout), BotAi(ort.models), listener)
                e.settings = BotSettings(autoPlay = false, continuous = true, clickPopups = false,
                    showCards = false, cardConfidence = 0.76)
                engine = e
                e.start()
            } catch (t: Throwable) {
                publish(snapshot(status = "模型加载失败：${t.javaClass.simpleName}", recommendation = "不可用", remaining = "请使用含模型的构建"))
            } finally { starting.set(false) }
        }, "assistant-engine-start").start()
    }

    private val listener = object : BotListener {
        override fun onStatus(text: String) = update { copy(it, status = text) }
        override fun onRole(position: Position?) = update { copy(it, status = "${textStatus(it)} · ${position?.label ?: "角色未知"}") }
        override fun onHand(cards: String) = update { copy(it, hand = cards) }
        override fun onThreeCards(cards: String) = update { copy(it, bottom = cards) }
        override fun onSuggestion(text: String) = update { copy(it, recommendation = text.ifEmpty { "--" }) }
        override fun onCandidates(text: String) = update { copy(it, candidates = text.ifEmpty { "--" }) }
        override fun onLeftPlayed(text: String) = update { copy(it, left = text) }
        override fun onRightPlayed(text: String) = update { copy(it, right = text) }
        override fun onRemaining(counts: IntArray) = update { copy(it, remaining = remainingText(counts)) }
        override fun onNewGame() = publish(snapshot(status = "等待新一局"))
        override fun onStopped() = update { copy(it, status = "分析已停止") }
    }

    private fun remainingText(counts: IntArray): String {
        val labels = Cards.REAL_DESC.map { if (it == 'T') "10" else it.toString() }.joinToString(" ")
        return labels + "\n" + counts.joinToString("  ")
    }

    private fun textStatus(s: AssistantSnapshot) = s.status().substringBefore(" · ")
    private inline fun update(block: (AssistantSnapshot) -> AssistantSnapshot) { publish(block(state.get())) }
    private fun copy(s: AssistantSnapshot, status: String = s.status(), hand: String = s.hand(), bottom: String = s.bottom(),
                     left: String = s.leftPlay(), right: String = s.rightPlay(), recommendation: String = s.recommendation(),
                     candidates: String = s.candidates(), remaining: String = s.remaining()) =
        AssistantSnapshot(status, hand, bottom, left, right, recommendation, candidates, remaining)
    private fun snapshot(status: String, hand: String = "", bottom: String = "", left: String = "", right: String = "",
                         recommendation: String = "--", candidates: String = "--", remaining: String = "等待稳定识别") =
        AssistantSnapshot(status, hand, bottom, left, right, recommendation, candidates, remaining)
    private fun publish(snapshot: AssistantSnapshot) { state.set(snapshot) }
    @JvmStatic fun current(): AssistantSnapshot = state.get()
}
