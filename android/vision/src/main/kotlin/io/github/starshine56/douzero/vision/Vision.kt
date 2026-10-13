package io.github.starshine56.douzero.vision

import io.github.starshine56.douzero.core.Cards
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.roundToInt

/** A template match in work-image coordinates. */
data class Match(val x: Int, val y: Int, val w: Int, val h: Int, val score: Double) {
    val cx: Int get() = x + w / 2
    val cy: Int get() = y + h / 2
}

/** One recognised card: [rank] is a real-card char (3..9TJQKA2XD). */
data class CardHit(val rank: Char, val match: Match)

/** Recognised cards, left to right. */
class CardRow(val hits: List<CardHit>) {
    val text: String get() = hits.joinToString("") { it.rank.toString() }
    val size: Int get() = hits.size
    fun isEmpty() = hits.isEmpty()
    fun isNotEmpty() = hits.isNotEmpty()
    fun env(): List<Int> = Cards.parse(text)

    /** Upstream order (D X 2 A ... 3). */
    fun sortedText(): String = if (hits.isEmpty()) "" else Cards.sortDesc(text)
}

/**
 * Holds the latest screenshot scaled to height 810 (the "work" image) and
 * runs the template matching on it.
 */
class Vision(val templates: TemplateStore, @Volatile var layout: Layout) {
    private val device = Mat()
    val work = Mat()
    private val workGray = Mat()
    private var grayValid = false

    /** Device pixels per work pixel. */
    var scale = 1.0
        private set

    /** Grab a new frame. Returns false when there is no (landscape) frame. */
    fun capture(screen: ScreenSource): Boolean {
        if (!screen.capture(device) || device.empty()) return false
        if (device.cols() < device.rows()) return false
        setDeviceImage(device)
        return true
    }

    /** Use [bgr] (device resolution, landscape) as the current frame. */
    fun setDeviceImage(bgr: Mat) {
        val h = bgr.rows()
        val f = Layout.REF_H.toDouble() / h
        val w = (bgr.cols() * f).roundToInt()
        if (h == Layout.REF_H) bgr.copyTo(work)
        else Imgproc.resize(bgr, work, Size(w.toDouble(), Layout.REF_H.toDouble()), 0.0, 0.0,
            if (f < 1) Imgproc.INTER_AREA else Imgproc.INTER_LINEAR)
        scale = h.toDouble() / Layout.REF_H
        grayValid = false
    }

    /** Use an image that is already in work scale (e.g. a saved calibration screenshot). */
    fun setWorkImage(bgr: Mat, deviceScale: Double = 1.0) {
        bgr.copyTo(work)
        scale = deviceScale
        grayValid = false
    }

    private fun gray(): Mat {
        if (!grayValid) {
            Imgproc.cvtColor(work, workGray, Imgproc.COLOR_BGR2GRAY)
            grayValid = true
        }
        return workGray
    }

    val width: Int get() = work.cols()

    fun rect(region: String): Rect = clamp(Layout.toWork(layout[region], width))

    private fun clamp(r: Rect): Rect {
        val x = r.x.coerceIn(0, maxOf(0, work.cols() - 1))
        val y = r.y.coerceIn(0, maxOf(0, work.rows() - 1))
        val w = r.width.coerceAtMost(work.cols() - x).coerceAtLeast(0)
        val h = r.height.coerceAtMost(work.rows() - y).coerceAtLeast(0)
        return Rect(x, y, w, h)
    }

    /** Work coordinates -> device (screen) coordinates. */
    fun toDevice(x: Number, y: Number): Pair<Float, Float> =
        (x.toDouble() * scale).toFloat() to (y.toDouble() * scale).toFloat()

    private inline fun <T> withMatch(image: Mat, region: Rect, template: Mat, block: (FloatArray, Int, Int) -> T): T? {
        if (region.width < template.cols() || region.height < template.rows()) return null
        val roi = image.submat(region)
        val res = Mat()
        try {
            Imgproc.matchTemplate(roi, template, res, Imgproc.TM_CCOEFF_NORMED)
            val cols = res.cols()
            val rows = res.rows()
            val data = FloatArray(cols * rows)
            res.get(0, 0, data)
            return block(data, cols, rows)
        } finally {
            res.release()
            roi.release()
        }
    }

    /** Best match of [name] inside [region] if its score reaches [confidence] (upstream LocateOnImage). */
    fun locate(name: String, region: String, confidence: Double = 0.8): Match? =
        locateIn(name, rect(region), confidence)

    fun locateIn(name: String, r: Rect, confidence: Double = 0.8): Match? {
        val t = templates[name] ?: return null
        return withMatch(work, r, t) { data, cols, _ ->
            var best = -2f
            var bi = -1
            for (i in data.indices) if (data[i] > best) { best = data[i]; bi = i }
            if (bi < 0 || best.isNaN() || best < confidence) null
            else Match(r.x + bi % cols, r.y + bi / cols, t.cols(), t.rows(), best.toDouble())
        }
    }

    /**
     * Recognise the cards in [region] with template set [set] (m/o/c/z).
     * Like upstream find_cards, but matched in grey (red and black glyphs of a
     * rank look the same) and every rank competes for each position
     * (non-maximum suppression across ranks) instead of being counted separately.
     */
    fun cards(set: Char, region: String, confidence: Double = 0.8, jokerByColor: Boolean = true): CardRow {
        val r = rect(region)
        val candidates = ArrayList<CardHit>()
        // Public screenshots from 欢乐/JJ/途游 clients show three common glyph scales.
        // Width is a stable discriminator after every screenshot is normalized to height 810.
        val glyphScale = when {
            width <= 1520 -> 0.90
            width >= 1840 -> 1.15
            else -> 1.00
        }
        for (rank in Cards.REAL_DESC) {
            val t = templates.grayScaled("$set$rank", glyphScale) ?: continue
            withMatch(gray(), r, t) { data, cols, rows ->
                for (yy in 0 until rows) for (xx in 0 until cols) {
                    val v = data[yy * cols + xx]
                    if (v >= confidence) candidates += CardHit(rank, Match(r.x + xx, r.y + yy, t.cols(), t.rows(), v.toDouble()))
                }
            }
        }
        candidates.sortByDescending { it.match.score }
        val kept = ArrayList<CardHit>()
        for (c in candidates) {
            val m = c.match
            val minDx = maxOf(12, (m.w * 0.6).roundToInt())
            // One glyph can produce a second hit lower down (notably JOKER letters). Suppress by x
            // across a generous y range; genuinely selected cards move as a whole and still yield one hit.
            val clash = kept.any { abs(it.match.x - m.x) <= minDx && abs(it.match.y - m.y) <= m.h * 1.6 }
            if (!clash) kept += c
        }
        var hits = kept.sortedBy { it.match.x }
        if (jokerByColor) hits = resolveJokers(hits)
        return CardRow(hits)
    }

    /**
     * The three landlord cards: [cards] with set 'z', plus jokers no template
     * matched. A joker looks different from the rank glyphs there, so ink in
     * the glyph band that no recognised card explains counts as a joker, big
     * (red ink) or small (black ink).
     */
    fun landlordCards(confidence: Double = 0.8): CardRow {
        val row = cards('z', Regions.LANDLORD_CARDS, confidence)
        if (row.size !in 1..2) return row
        val r = rect(Regions.LANDLORD_CARDS)
        val y0 = row.hits.minOf { it.match.y }
        val y1 = row.hits.maxOf { it.match.y + it.match.h }.coerceAtMost(work.rows())
        val covered = BooleanArray(r.width)
        for (h in row.hits) for (x in h.match.x - 3..h.match.x + h.match.w + 2) {
            if (x - r.x in covered.indices) covered[x - r.x] = true
        }
        val band = work.submat(Rect(r.x, y0, r.width, y1 - y0))
        val px = ByteArray(band.rows() * band.cols() * 3)
        val cont = band.clone()
        cont.get(0, 0, px)
        cont.release()
        band.release()
        // ink per column: red / dark pixel counts
        val red = IntArray(r.width)
        val dark = IntArray(r.width)
        for (yy in 0 until y1 - y0) for (xx in 0 until r.width) {
            if (covered[xx]) continue
            val i = (yy * r.width + xx) * 3
            val b = px[i].toInt() and 0xff
            val g = px[i + 1].toInt() and 0xff
            val rr = px[i + 2].toInt() and 0xff
            if (rr > 120 && rr - g > 60 && rr - b > 60) red[xx]++
            else if (maxOf(rr, g, b) < 100) dark[xx]++
        }
        // group inked columns into blobs (gaps up to 3 columns)
        val blobs = ArrayList<Triple<Int, Int, Double>>() // start, end, red share
        var x = 0
        while (x < r.width) {
            if (red[x] + dark[x] == 0) { x++; continue }
            var end = x
            var gap = 0
            var k = x
            while (k < r.width && gap <= 3) {
                if (red[k] + dark[k] > 0) { end = k; gap = 0 } else gap++
                k++
            }
            val ink = (x..end).sumOf { red[it] + dark[it] }
            // a blob at the edge of the area is a card cut off by it, not a joker
            val atEdge = x == 0 || end >= r.width - 2
            if (ink >= 30 && end - x + 1 >= 5 && !atEdge) {
                blobs += Triple(x, end, (x..end).sumOf { red[it] }.toDouble() / ink)
            }
            x = end + 1
        }
        val extra = blobs.sortedByDescending { it.second - it.first }.take(3 - row.size)
        if (extra.isEmpty()) return row
        val ranks = if (extra.size == 2) {
            val redder = extra.maxBy { it.third }
            extra.map { if (it === redder) 'D' else 'X' }
        } else {
            extra.map { if (it.third > 0.5) 'D' else 'X' }
        }
        val jokers = extra.mapIndexed { i, b ->
            CardHit(ranks[i], Match(r.x + b.first, y0, b.second - b.first + 1, y1 - y0, 0.0))
        }
        return CardRow((row.hits + jokers).sortedBy { it.match.x })
    }

    /** Both jokers share the "J" glyph; the letters under it are red on the big joker and black on the small one. */
    private fun resolveJokers(hits: List<CardHit>): List<CardHit> {
        val jokerIdx = hits.indices.filter { hits[it].rank == 'X' || hits[it].rank == 'D' }
        if (jokerIdx.isEmpty()) return hits
        val red = jokerIdx.associateWith { jokerRedShare(hits[it].match) }
        val rank = HashMap<Int, Char>()
        if (jokerIdx.size >= 2) {
            // never more than one of each: the redder one is the big joker
            val byRed = jokerIdx.sortedByDescending { red.getValue(it) }
            rank[byRed[0]] = 'D'
            for (i in byRed.drop(1)) rank[i] = 'X'
        } else {
            rank[jokerIdx[0]] = if (red.getValue(jokerIdx[0]) > 0.5) 'D' else 'X'
        }
        return hits.indices.map { i -> rank[i]?.let { hits[i].copy(rank = it) } ?: hits[i] }
    }

    /** Share of red among the ink pixels right below a joker's "J" (0 when there is no ink). */
    fun jokerRedShare(m: Match): Double {
        val y0 = (m.y + m.h).coerceAtMost(work.rows())
        val y1 = (m.y + 2 * m.h).coerceAtMost(work.rows())
        val x1 = (m.x + m.w).coerceAtMost(work.cols())
        if (y1 - y0 < 2 || x1 - m.x < 2) return 0.0
        val crop = work.submat(Rect(m.x, y0, x1 - m.x, y1 - y0))
        val px = ByteArray(crop.rows() * crop.cols() * 3)
        val cont = crop.clone()
        cont.get(0, 0, px)
        cont.release()
        crop.release()
        var red = 0
        var dark = 0
        for (i in px.indices step 3) {
            val b = px[i].toInt() and 0xff
            val g = px[i + 1].toInt() and 0xff
            val r = px[i + 2].toInt() and 0xff
            if (r > 120 && r - g > 60 && r - b > 60) red++
            else if (maxOf(r, g, b) < 100) dark++
        }
        return if (red + dark == 0) 0.0 else red.toDouble() / (red + dark)
    }

    /** Copy of a region, for change detection. */
    fun snapshot(region: String): Mat {
        val m = Mat()
        val roi = work.submat(rect(region))
        roi.copyTo(m)
        roi.release()
        return m
    }

    companion object {
        /** Mean absolute grey difference between two snapshots (0..255). */
        fun difference(a: Mat, b: Mat): Double {
            if (a.size() != b.size() || a.empty()) return 255.0
            val ga = Mat()
            val gb = Mat()
            val d = Mat()
            try {
                Imgproc.cvtColor(a, ga, Imgproc.COLOR_BGR2GRAY)
                Imgproc.cvtColor(b, gb, Imgproc.COLOR_BGR2GRAY)
                Core.absdiff(ga, gb, d)
                return Core.mean(d).`val`[0]
            } finally {
                ga.release(); gb.release(); d.release()
            }
        }
    }
}

/** Supplies screenshots: BGR, full device resolution. */
fun interface ScreenSource {
    /** Fill [dst] with the latest screen frame; false if none is available. */
    fun capture(dst: Mat): Boolean
}

/** Performs taps in device (screen) coordinates. */
fun interface Tapper {
    fun tap(x: Float, y: Float)
}
