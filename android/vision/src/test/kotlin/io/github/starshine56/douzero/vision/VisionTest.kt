package io.github.starshine56.douzero.vision

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Recognition on real phone screenshots (bot/src/test/resources/screens,
 * 20:9 screen scaled to height 810; player avatars and names blanked out).
 */
class VisionTest {
    private val dir = File(System.getProperty("templates.dir"))
    private val store = TemplateStore { name -> File(dir, "$name.png").takeIf { it.exists() }?.readBytes() }

    @BeforeTest
    fun loadOpenCv() {
        nu.pattern.OpenCV.loadLocally()
    }

    private fun screen(name: String): Mat {
        val path = File(javaClass.getResource("/screens/$name.jpg")!!.toURI()).path
        return Imgcodecs.imread(path).also { assertTrue(!it.empty(), name) }
    }

    private fun vision(name: String): Vision = Vision(store, Layout()).apply { setDeviceImage(screen(name)) }

    private fun found(v: Vision, name: String, region: String = Regions.BUTTONS) = v.locate(name, region) != null

    @Test
    fun dealtHandBeforeBidding() {
        val v = vision("deal")
        assertEquals(1782, v.width)
        assertEquals("X222QQTT986655544", v.cards('m', Regions.MY_HAND).sortedText())
        assertTrue(v.cards('z', Regions.LANDLORD_CARDS).isEmpty())
        assertNotNull(v.locate("chat", Regions.CHAT))
        for (b in listOf("jiaodizhu_btn", "qiangdizhu_btn", "jiabei_btn", "play_card", "pass_btn")) {
            assertTrue(!found(v, b), b)
        }
    }

    @Test
    fun robbingStage() {
        val v = vision("rob")
        assertEquals("X222QQTT986655544", v.cards('m', Regions.MY_HAND).sortedText())
        assertTrue(found(v, "qiangdizhu_btn"))
        assertTrue(found(v, "buqiang_btn"))
        // "地主": the same template finds the 叫地主 button, so the engine checks 抢地主 first
        assertTrue(found(v, "jiaodizhu_btn"))
        assertTrue(!found(v, "jiabei_btn"))
        assertTrue(v.cards('o', Regions.LEFT_PLAYED).isEmpty())
        assertTrue(v.cards('o', Regions.RIGHT_PLAYED).isEmpty())
    }

    @Test
    fun doublingStage() {
        val v = vision("double")
        assertEquals("KT3", v.cards('z', Regions.LANDLORD_CARDS).sortedText())
        val jiabei = assertNotNull(v.locate("jiabei_btn", Regions.BUTTONS, 0.9))
        assertTrue(jiabei.x < 700, "加倍, not 超级加倍")
        assertTrue(found(v, "chaojijiabei_btn"))
        assertTrue(found(v, "bujiabei_btn"))
        assertTrue(!found(v, "qiangdizhu_btn"))
        assertTrue(!found(v, "play_card"))
    }

    @Test
    fun ourTurnToPlay() {
        val v = vision("my_turn")
        assertEquals("X222QQTT986655544", v.cards('m', Regions.MY_HAND).sortedText())
        assertEquals("33", v.cards('o', Regions.LEFT_PLAYED).sortedText())
        assertTrue(v.cards('o', Regions.RIGHT_PLAYED).isEmpty())
        assertTrue(v.cards('c', Regions.MY_PLAYED).isEmpty())
        // the played 3 is greyed out in the 记牌器 bar but still reads
        assertEquals("KT3", v.cards('z', Regions.LANDLORD_CARDS).sortedText())
        assertTrue(found(v, "play_card"))
        assertTrue(found(v, "pass_btn"))
        assertNull(v.locate("buchu", Regions.MY_PASS))
        assertNull(v.locate("buchu", Regions.LEFT_PASS))
        assertNull(v.locate("buchu", Regions.RIGHT_PASS))
    }

    @Test
    fun passLabelsNextToPlayers() {
        // 不叫 / 不加倍 share the "不" of 不出
        assertNotNull(vision("rob").locate("buchu", Regions.RIGHT_PASS))
        val v = vision("double")
        assertNotNull(v.locate("buchu", Regions.LEFT_PASS))
        assertNotNull(v.locate("buchu", Regions.RIGHT_PASS))
        assertNull(v.locate("buchu", Regions.MY_PASS))
    }

    @Test
    fun selectedCardsSitHigher() {
        val hits = vision("selected").cards('m', Regions.MY_HAND).hits
        assertEquals(17, hits.size)
        val base = hits.maxOf { it.match.y }
        val popped = hits.filter { it.match.y < base - 20 }.map { it.rank }.joinToString("")
        assertEquals("44", popped)
    }

    @Test
    fun jokerColour() {
        val img = screen("my_turn")
        val v = Vision(store, Layout())
        v.setDeviceImage(img)
        val joker = v.cards('m', Regions.MY_HAND).hits.first()
        assertEquals('X', joker.rank)
        // paint the joker's letters red: the big joker
        val card = Rect(joker.match.x - 6, joker.match.y, 60, 210)
        val roi = img.submat(card)
        val grey = Mat()
        org.opencv.imgproc.Imgproc.cvtColor(roi, grey, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY)
        val dark = Mat()
        org.opencv.core.Core.compare(grey, Scalar(110.0), dark, org.opencv.core.Core.CMP_LT)
        roi.setTo(Scalar(40.0, 40.0, 210.0), dark)
        v.setDeviceImage(img)
        assertEquals("D222QQTT986655544", v.cards('m', Regions.MY_HAND).sortedText())
    }

    /** Replace the landlord card "3" with a joker-like glyph no template knows, in [color]. */
    private fun jokerInLandlordCards(color: Scalar): Vision {
        val img = screen("double")
        val joker = Imgcodecs.imread(File(dir, "mX.png").path)
        val small = Mat()
        org.opencv.imgproc.Imgproc.resize(joker, small, org.opencv.core.Size(16.0, 30.0))
        val grey = Mat()
        org.opencv.imgproc.Imgproc.cvtColor(small, grey, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY)
        val ink = Mat()
        org.opencv.core.Core.compare(grey, Scalar(120.0), ink, org.opencv.core.Core.CMP_LT)
        small.setTo(Scalar(245.0, 245.0, 245.0))
        small.setTo(color, ink)
        img.submat(Rect(864, 20, 26, 36)).setTo(Scalar(245.0, 245.0, 245.0))
        small.copyTo(img.submat(Rect(868, 23, 16, 30)))
        return Vision(store, Layout()).apply { setDeviceImage(img) }
    }

    @Test
    fun longStraightsAndCannotBeat() {
        val v = vision("yaobuqi")
        // the big joker's letters are red
        assertEquals("D2AAKQQQTT98773", v.cards('m', Regions.MY_HAND).sortedText())
        // 8-card straights on both sides, the left one starts further out than short moves
        assertEquals("QJT98765", v.cards('o', Regions.LEFT_PLAYED).sortedText())
        assertEquals("T9876543", v.cards('o', Regions.RIGHT_PLAYED).sortedText())
        assertEquals("K84", v.landlordCards().sortedText())
        assertTrue(found(v, "yaobuqi"))
        assertTrue(!found(v, "play_card"))
        assertTrue(v.cards('c', Regions.MY_PLAYED).isEmpty())
    }

    @Test
    fun ourPlayedCards() {
        val v = vision("my_played")
        assertEquals("55", v.cards('c', Regions.MY_PLAYED).sortedText())
        assertTrue(v.cards('o', Regions.LEFT_PLAYED).isEmpty())
        assertTrue(!found(v, "play_card"))
    }

    @Test
    fun bigJokerInHandWhileBidding() {
        val v = vision("deal_joker")
        assertEquals("D2AAKQQQTT9877553", v.cards('m', Regions.MY_HAND).sortedText())
        assertNotNull(v.locate("buchu", Regions.RIGHT_PASS))
        assertTrue(v.landlordCards().isEmpty())
    }

    @Test
    fun withoutCardCounter() {
        // no 记牌器: the landlord cards sit alone at the top centre
        val v = vision("no_counter")
        assertEquals("KJ9", v.landlordCards().sortedText())
        assertEquals("QQT98886554433", v.cards('m', Regions.MY_HAND).sortedText())
        // the 不出 labels of the left player and ours
        assertNotNull(v.locate("buchu", Regions.LEFT_PASS))
        assertNotNull(v.locate("buchu", Regions.MY_PASS))
        assertNull(v.locate("buchu", Regions.RIGHT_PASS))
        assertTrue(v.cards('c', Regions.MY_PLAYED).isEmpty())
        assertTrue(v.cards('o', Regions.LEFT_PLAYED).isEmpty())
    }

    @Test
    fun alarmClockShowsWhoIsThinking() {
        val expected = mapOf(
            "deal" to Regions.CLOCK_RIGHT, "rob" to Regions.CLOCK_ME, "double" to Regions.CLOCK_ME,
            "my_turn" to Regions.CLOCK_ME, "selected" to Regions.CLOCK_ME, "double_joker" to Regions.CLOCK_ME,
            "yaobuqi" to Regions.CLOCK_ME, "my_played" to Regions.CLOCK_LEFT, "deal_joker" to Regions.CLOCK_LEFT,
            "no_counter" to Regions.CLOCK_RIGHT,
        )
        for ((screen, at) in expected) {
            val v = vision(screen)
            for (r in listOf(Regions.CLOCK_LEFT, Regions.CLOCK_RIGHT, Regions.CLOCK_ME)) {
                val name = if (r == Regions.CLOCK_ME) "clock_me" else "clock"
                assertEquals(r == at, v.locate(name, r) != null, "$screen: $r")
            }
        }
    }

    @Test
    fun endOfGameScreens() {
        val result = vision("result")
        assertNotNull(result.locate("continue", Regions.CONTINUE))
        assertNotNull(result.locate("lose", Regions.RESULT))
        assertNull(result.locate("fangqi", Regions.FANGQI))
        // 返豆暴击: press 放弃 (the engine checks pop-ups before 继续游戏, which still shows dimmed)
        assertNotNull(vision("popup").locate("fangqi", Regions.FANGQI))
        // 赢豆暴击 after a win, 星星保护 after a loss
        assertNotNull(vision("win_popup").locate("fangqi_win", Regions.FANGQI))
        assertNotNull(vision("star_protect").locate("fangqi_star", Regions.FANGQI))
        for (s in listOf("result", "popup", "win_popup", "star_protect")) {
            val v = vision(s)
            val found = listOf("fangqi", "fangqi_win", "fangqi_star").filter { v.locate(it, Regions.FANGQI) != null }
            assertTrue(found.size <= 1, "$s: $found")
        }
    }

    @Test
    fun bigJokerAmongLandlordCards() {
        // real screenshot: the big joker is a narrow card with red "JOKER" letters
        val v = vision("double_joker")
        assertEquals("2AAKK987766654433", v.cards('m', Regions.MY_HAND).sortedText())
        assertEquals("J8", v.cards('z', Regions.LANDLORD_CARDS).sortedText())
        assertEquals("DJ8", v.landlordCards().sortedText())
        assertNotNull(v.locate("jiabei_btn", Regions.BUTTONS, 0.9))
        assertNotNull(v.locate("buchu", Regions.RIGHT_PASS))
    }

    @Test
    fun jokersAmongLandlordCards() {
        assertEquals("KT3", vision("double").landlordCards().sortedText())
        assertEquals("DKT", jokerInLandlordCards(Scalar(40.0, 40.0, 210.0)).landlordCards().sortedText())
        assertEquals("XKT", jokerInLandlordCards(Scalar(30.0, 30.0, 30.0)).landlordCards().sortedText())
    }

    @Test
    fun noPopupFalsePositives() {
        // the pop-up templates are still the PC ones: they must not fire on the phone table
        val other = listOf("continue" to Regions.CONTINUE, "win" to Regions.RESULT, "lose" to Regions.RESULT,
            "over" to Regions.BEANS_LEFT, "over" to Regions.BEANS_ME, "over" to Regions.BEANS_RIGHT,
            "mingpai_btn" to Regions.BUTTONS, "ming_btn" to Regions.BUTTONS, "laotou" to Regions.LANDLORD_CARDS)
        for (s in listOf("deal", "rob", "double", "my_turn", "selected", "double_joker", "yaobuqi", "my_played", "deal_joker", "no_counter")) {
            val v = vision(s)
            for ((name, region) in other) assertNull(v.locate(name, region), "$s: $name")
        }
    }

    @Test
    fun widerScreensUseAnchors() {
        // a wider phone: centre content moves by half the extra width, right-edge content by all of it
        val shot = screen("my_turn")
        val extra = 100
        val img = Mat(810, shot.cols() + extra, CvType.CV_8UC3, Scalar(150.0, 90.0, 40.0))
        shot.submat(Rect(0, 0, 900, 810)).copyTo(img.submat(Rect(0, 0, 900, 810)))
        shot.submat(Rect(900, 0, shot.cols() - 900, 810)).copyTo(img.submat(Rect(900 + extra, 0, shot.cols() - 900, 810)))
        val hand = shot.submat(Rect(200, 480, 1400, 140))
        img.submat(Rect(200, 480, 1500, 140)).setTo(Scalar(150.0, 90.0, 40.0))
        hand.copyTo(img.submat(Rect(200 + extra / 2, 480, 1400, 140)))
        val v = Vision(store, Layout())
        v.setWorkImage(img)
        assertEquals("X222QQTT986655544", v.cards('m', Regions.MY_HAND).sortedText())
        assertEquals("33", v.cards('o', Regions.LEFT_PLAYED).sortedText())
        assertNotNull(v.locate("chat", Regions.CHAT))

        val r = v.rect(Regions.LEFT_PLAYED)
        assertEquals(Regions.DEFAULTS[Regions.LEFT_PLAYED], Layout.fromWork(r, Anchor.LEFT, v.width))
    }

    @Test
    fun deviceImageIsScaledToWorkHeight() {
        val dev = Mat(1080, 2400, CvType.CV_8UC3, Scalar(140.0, 95.0, 35.0))
        val v = Vision(store, Layout())
        v.setDeviceImage(dev)
        assertEquals(810, v.work.rows())
        assertEquals(1800, v.work.cols())
        val (x, y) = v.toDevice(900, 405)
        assertEquals(1200f, x, 0.01f)
        assertEquals(540f, y, 0.01f)
    }

    @Test
    fun bundledTemplatesDecode() {
        val missing = TemplateStore.allNames().filter { store[it] == null }
        assertTrue(missing.isEmpty(), "missing templates: $missing")
        Imgcodecs.imread(File(dir, "m3.png").path).let { assertTrue(!it.empty()) }
    }
}
