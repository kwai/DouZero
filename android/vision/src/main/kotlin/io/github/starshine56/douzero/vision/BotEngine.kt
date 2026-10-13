package io.github.starshine56.douzero.vision

import io.github.starshine56.douzero.core.BidAdvisor
import io.github.starshine56.douzero.core.Cards
import io.github.starshine56.douzero.core.DoubleChoice
import io.github.starshine56.douzero.core.DouZeroAgent
import io.github.starshine56.douzero.core.GameState
import io.github.starshine56.douzero.core.ModelSet
import io.github.starshine56.douzero.core.MoveTypes
import io.github.starshine56.douzero.core.Position
import io.github.starshine56.douzero.core.Thresholds
import io.github.starshine56.douzero.core.moveType
import java.util.Locale

data class BotSettings(
    val thresholds: Thresholds = Thresholds(),
    /** true: tap automatically; false: only show suggestions (upstream 手动模式). */
    val autoPlay: Boolean = false,
    /** Keep playing the next game (upstream 自动/单局 toggle). */
    val continuous: Boolean = true,
    /** Close pop-ups / press 继续游戏, 快速开始 ... in auto mode. */
    val clickPopups: Boolean = false,
    /** 明牌 when the hand is very strong (upstream default behaviour). */
    val showCards: Boolean = false,
    val cardConfidence: Double = 0.8,
    /** Tap this many work pixels below the centre of a card's rank glyph. */
    val tapOffsetY: Int = 30,
    /** Extra pause before each automatic move, in ms. */
    val thinkDelayMs: Long = 0,
)

/** UI callbacks; called from the bot thread. */
interface BotListener {
    fun onStatus(text: String) {}
    fun onLog(text: String) {}
    fun onRole(position: Position?) {}
    fun onHand(cards: String) {}
    fun onThreeCards(cards: String) {}
    fun onBidScore(text: String) {}
    fun onPreScore(text: String) {}
    fun onSuggestion(text: String) {}
    fun onCandidates(text: String) {}
    fun onLeftPlayed(text: String) {}
    fun onRightPlayed(text: String) {}

    /** Unseen cards per rank, index-aligned with [Cards.REAL_DESC] (记牌器). */
    fun onRemaining(counts: IntArray) {}
    fun onNewGame() {}
    fun onStopped() {}
}

/** The AI side: DouZero card play + bid / double networks. */
class BotAi(models: ModelSet) {
    val agent = DouZeroAgent(models)
    val advisor = BidAdvisor(models, agent)
}

/**
 * The automation loop, ported from the upstream Worker thread (main.py):
 * wait for a game, bid / rob / double, then follow every move on the table
 * and play the DouZero move whenever it is our turn.
 */
class BotEngine(
    private val screen: ScreenSource,
    private val tapper: Tapper,
    val vision: Vision,
    private val ai: BotAi,
    private val listener: BotListener,
) {
    @Volatile
    var settings = BotSettings()

    @Volatile
    private var running = false
    private var thread: Thread? = null
    private var lastStatus = ""
    private var playedThisRun = false

    /** After a finished game only a fresh deal counts as the next game. */
    private var needFreshDeal = false

    /** Landlord cards found during the doubling stage of the current game. */
    private var knownThree: List<Int>? = null

    /** Our 17 cards while bidding: once we are the landlord, the new cards are the landlord cards. */
    private var bidHand: List<Int>? = null

    val isRunning: Boolean get() = running

    fun start() {
        if (running) return
        running = true
        playedThisRun = false
        needFreshDeal = false
        thread = Thread(::runLoop, "hlddz-bot").apply {
            isDaemon = true
            start()
        }
    }

    fun stop() {
        running = false
        thread?.interrupt()
    }

    private class Stopped : RuntimeException()
    private class GameEnded(message: String) : RuntimeException(message)

    private enum class Seat(
        val played: String, val pass: String, val set: Char, val label: String,
        val clock: String, val clockRegion: String,
    ) {
        ME(Regions.MY_PLAYED, Regions.MY_PASS, 'c', "自己", "clock_me", Regions.CLOCK_ME),
        RIGHT(Regions.RIGHT_PLAYED, Regions.RIGHT_PASS, 'o', "下家", "clock", Regions.CLOCK_RIGHT),
        LEFT(Regions.LEFT_PLAYED, Regions.LEFT_PASS, 'o', "上家", "clock", Regions.CLOCK_LEFT),
    }

    // ---------------------------------------------------------------- helpers

    private val conf: Double get() = settings.cardConfidence

    private fun sleep(ms: Long) {
        if (!running) throw Stopped()
        try {
            Thread.sleep(ms)
        } catch (e: InterruptedException) {
            throw Stopped()
        }
        if (!running) throw Stopped()
    }

    private fun now() = System.currentTimeMillis()

    private fun log(s: String) = listener.onLog(s)

    private fun status(s: String) {
        if (s == lastStatus) return
        lastStatus = s
        listener.onStatus(s)
        log(s)
    }

    /** Status line only (not logged): what a wait loop is looking at right now. */
    private fun hint(s: String) {
        if (s == lastStatus) return
        lastStatus = s
        listener.onStatus(s)
    }

    private fun fmt(v: Double) = String.format(Locale.ROOT, "%.3f", v)

    /** Grab a fresh frame into [vision]; waits while the screen is unavailable or portrait. */
    private fun grab() {
        while (true) {
            if (!running) throw Stopped()
            if (vision.capture(screen)) return
            status("等待屏幕画面（请横屏打开游戏）")
            sleep(500)
        }
    }

    private fun tapWork(x: Int, y: Int) {
        val (dx, dy) = vision.toDevice(x, y)
        tapper.tap(dx, dy)
    }

    private fun tapMatch(m: Match) = tapWork(m.cx, m.cy)

    private fun clickTemplate(name: String, region: String, confidence: Double = 0.8): Boolean {
        grab()
        val m = vision.locate(name, region, confidence) ?: return false
        tapMatch(m)
        return true
    }

    private fun cards(set: Char, region: String): CardRow {
        val threshold = when (set) {
            'o' -> maxOf(conf, 0.80) // table cards are isolated and should stay strict
            'z' -> minOf(conf, 0.68) // supplied independent-app bottom cards use a smaller font
            else -> conf
        }
        return vision.cards(set, region, threshold)
    }

    /** Port of detect_start_btn: notice the end of a game and handle pop-ups. */
    private fun checkMeta() {
        for (r in listOf(Regions.BEANS_LEFT, Regions.BEANS_ME, Regions.BEANS_RIGHT)) {
            if (vision.locate("over", r, 0.9) != null) throw GameEnded("豆子出现，对局结束")
        }
        val s = settings
        if (!s.autoPlay) return
        // pop-ups first: 继续游戏 may still match dimmed underneath one
        if (s.clickPopups) {
            for ((name, region) in POPUPS) {
                val m = vision.locate(name, region) ?: continue
                log("点击弹窗：$name")
                if (name == "chacha") tapWork(m.x + 20, m.y + 20) else tapMatch(m)
                sleep(1000)
                grab()
            }
        }
        vision.locate("continue", Regions.CONTINUE)?.let { m ->
            if (s.continuous && s.clickPopups) tapMatch(m)
            throw GameEnded("继续游戏")
        }
        if (vision.locate("win", Regions.RESULT) != null || vision.locate("lose", Regions.RESULT) != null) {
            throw GameEnded("胜负已分")
        }
    }

    // ---------------------------------------------------------------- main loop

    private fun runLoop() {
        log("机器人启动（" + (if (settings.autoPlay) "自动" else "手动提示") + "模式）")
        try {
            while (running) {
                try {
                    listener.onNewGame()
                    knownThree = null
                    bidHand = null
                    waitForGame()
                    biddingStage()
                    playStage()
                    needFreshDeal = true
                    if (!settings.continuous) {
                        status("单局模式：本局结束，已停止")
                        break
                    }
                } catch (e: GameEnded) {
                    needFreshDeal = true
                    log("对局结束：${e.message}")
                    listener.onNewGame()
                    if (!settings.continuous && playedThisRun) {
                        status("单局模式：本局结束，已停止")
                        break
                    }
                    sleep(2000)
                } catch (e: Stopped) {
                    throw e
                } catch (e: Exception) {
                    // the state of this game is lost: wait for the next deal instead of joining blindly
                    needFreshDeal = true
                    status("本局出错，等待下一局：$e")
                    log("本局出错：${e.stackTraceToString().take(600)}")
                    sleep(2000)
                }
            }
        } catch (_: Stopped) {
        } catch (t: Throwable) {
            log("出错已停止：${t.stackTraceToString().take(800)}")
            listener.onStatus("出错已停止：${t.message}")
        } finally {
            running = false
            lastStatus = ""
            listener.onStopped()
        }
    }

    /**
     * Port of before_start: wait for the table (chat button), then for the deal.
     * The PC version shows the landlord card backs while bidding; the phone
     * version shows nothing there, so a full hand without landlord cards
     * also counts as a new deal.
     */
    private fun waitForGame() {
        while (true) {
            grab()
            checkMeta()
            if (vision.locate("chat", Regions.CHAT) != null) break
            // the chat button may look different: any cards or table buttons also mean we are at a table
            val hand = cards('m', Regions.MY_HAND).size
            if (hand > 0 || TABLE_BUTTONS.any { vision.locate(it, Regions.BUTTONS) != null }) break
            val chat = vision.locate("chat", Regions.CHAT, 0.0)?.score ?: 0.0
            status("未进入游戏（画面 ${vision.width}x${vision.work.rows()}，聊天按钮相似度 ${String.format(Locale.ROOT, "%.1f", chat)}，手牌 0 张）")
            sleep(1000)
        }
        while (true) {
            grab()
            checkMeta()
            if (vision.locate("laotou", Regions.LANDLORD_CARDS) != null) break
            val hand = cards('m', Regions.MY_HAND).size
            if (hand >= 17 && cards('z', Regions.LANDLORD_CARDS).isEmpty()) break
            // started in the middle of a game: join it
            if (!needFreshDeal && hand > 0) break
            status("游戏未开始（识别到手牌 $hand 张）")
            sleep(500)
        }
        status("开始游戏")
    }

    private fun waitHand(vararg sizes: Int): CardRow {
        while (true) {
            grab()
            checkMeta()
            val row = cards('m', Regions.MY_HAND)
            if (row.size in sizes) return row
            hint("等待识别手牌：识别到 ${row.size} 张 ${row.sortedText()}")
            sleep(200)
        }
    }

    private fun waitThree(): List<Int> {
        while (true) {
            grab()
            checkMeta()
            val row = vision.landlordCards(minOf(conf, 0.68))
            if (row.size == 3) return row.env()
            hint("等待识别底牌：识别到 ${row.size} 张 ${row.sortedText()}")
            sleep(200)
        }
    }

    /** The landlord cards as the three cards our 20-card hand gained since bidding. */
    private fun threeFromHand(hand20: List<Int>): List<Int>? {
        val before = bidHand ?: return null
        val gained = hand20.toMutableList()
        if (!before.all { gained.remove(it) } || gained.size != 3) return null
        return gained.sorted()
    }

    // ---------------------------------------------------------------- bidding

    /** Port of choose_multiples_stage: 叫地主 / 抢地主 / 加倍 / 明牌. */
    private fun biddingStage() {
        grab()
        if (cards('z', Regions.LANDLORD_CARDS).isNotEmpty()) {
            log("底牌已翻开，跳过叫地主阶段")
            return
        }
        val t = settings.thresholds
        var haveBid = false
        var stolen = false
        status(if (settings.autoPlay) "自动叫地主/抢地主/加倍" else "叫地主阶段（仅提示）")

        // wait until some cards are dealt
        var hand = CardRow(emptyList())
        while (true) {
            grab()
            checkMeta()
            hand = cards('m', Regions.MY_HAND)
            if (hand.size >= 5) break
            sleep(50)
        }
        if (settings.autoPlay && settings.showCards && BidAdvisor.startWithShownCards(hand.env())) {
            vision.locate("ming_btn", Regions.BUTTONS)?.let {
                tapMatch(it)
                log("明牌开始")
            }
        }

        while (true) {
            var jiao: Match? = null
            var qiang: Match? = null
            var jiabei: Match? = null
            var waitedSince = now()
            while (true) {
                grab()
                checkMeta()
                jiao = vision.locate("jiaodizhu_btn", Regions.BUTTONS)
                qiang = vision.locate("qiangdizhu_btn", Regions.BUTTONS)
                jiabei = vision.locate("jiabei_btn", Regions.BUTTONS)
                if (jiao != null || qiang != null || jiabei != null) break
                // no doubling prompt for us although the landlord cards are open
                if (vision.landlordCards(minOf(conf, 0.68)).size == 3) {
                    if (now() - waitedSince > 8000) {
                        log("未出现加倍按钮，进入出牌阶段")
                        return
                    }
                } else {
                    waitedSince = now()
                }
                sleep(100)
            }

            if (jiabei == null) {
                hand = waitHand(17, 20)
                val cardList = hand.env()
                if (cardList.size == 17) bidHand = cardList
                val score = ai.advisor.bidScore(cardList)
                val farmer = ai.advisor.farmerScore(cardList)
                listener.onHand(hand.sortedText())
                listener.onBidScore("叫牌得分: ${fmt(score)}")
                listener.onPreScore("不叫得分: ${fmt(farmer)}")
                stolen = false
                // the phone 叫地主 template is just "地主", which 抢地主 contains too: check 抢 first
                if (qiang != null) {
                    stolen = true
                    val rob = ai.advisor.shouldRob(score, haveBid, t)
                    listener.onSuggestion(if (rob) "抢地主" else "不抢")
                    if (settings.autoPlay) {
                        log(if (rob) "抢地主" else "不抢地主")
                        clickTemplate(if (rob) "qiangdizhu_btn" else "buqiang_btn", Regions.BUTTONS)
                    }
                    haveBid = true
                } else if (jiao != null) {
                    haveBid = true
                    val call = ai.advisor.shouldCall(score, t)
                    listener.onSuggestion(if (call) "叫地主" else "不叫")
                    if (settings.autoPlay) {
                        log(if (call) "叫地主" else "不叫地主")
                        clickTemplate(if (call) "jiaodizhu_btn" else "bujiao_btn", Regions.BUTTONS)
                    }
                }
                sleep(if (settings.autoPlay) 1000 else 2000)
                continue
            }

            status("加倍阶段")
            hand = waitHand(17, 20)
            val cardList = hand.env()
            val isLandlord = cardList.size == 20
            val three = (if (isLandlord) threeFromHand(cardList) else null) ?: waitThree()
            knownThree = three
            val winRate = if (isLandlord) ai.advisor.landlordScore(cardList, three)
            else ai.advisor.farmerUpScore(cardList)
            listener.onThreeCards(Cards.formatDesc(three))
            listener.onHand(hand.sortedText())
            listener.onPreScore("局前得分: ${fmt(winRate)}")
            log("手牌 ${hand.sortedText()}，底牌 ${Cards.formatDesc(three)}，" +
                (if (isLandlord) "预估地主得分" else "预估农民得分") + " ${fmt(winRate)}")
            if (!settings.autoPlay) {
                grab()
                val superAvailable = vision.locate("chaojijiabei_btn", Regions.BUTTONS) != null
                val c = ai.advisor.doubleChoice(isLandlord, winRate, stolen, cardList, superAvailable, t)
                listener.onSuggestion("建议：${c.label}")
                return
            }
            log(if (isLandlord) "等待 2 秒，观察他人加倍情况" else "等待 3 秒，观察他人加倍情况")
            sleep(if (isLandlord) 2000 else 3000)
            grab()
            val superAvailable = vision.locate("chaojijiabei_btn", Regions.BUTTONS) != null
            val choice = ai.advisor.doubleChoice(isLandlord, winRate, stolen, cardList, superAvailable, t)
            log(choice.label)
            listener.onSuggestion(choice.label)
            when (choice) {
                DoubleChoice.SUPER -> clickTemplate("chaojijiabei_btn", Regions.BUTTONS)
                DoubleChoice.DOUBLE -> clickTemplate("jiabei_btn", Regions.BUTTONS, 0.9)
                DoubleChoice.NONE -> clickTemplate("bujiabei_btn", Regions.BUTTONS)
            }
            if (settings.showCards && ai.advisor.shouldShowCards(isLandlord, winRate, t)) {
                sleep(5000)
                if (clickTemplate("mingpai_btn", Regions.BUTTONS)) log("明牌")
            }
            return
        }
    }

    // ---------------------------------------------------------------- card play

    private fun findRole(): Position? {
        val n = cards('m', Regions.MY_HAND).size
        if (n == 20) return Position.LANDLORD
        if (n == 17) {
            // the landlord plays first; check the right side first like upstream
            if (cards('o', Regions.RIGHT_PLAYED).isNotEmpty()) return Position.LANDLORD_UP
            if (cards('o', Regions.LEFT_PLAYED).isNotEmpty()) return Position.LANDLORD_DOWN
        }
        return null
    }

    /** Port of init_cards + game_start. */
    private fun playStage() {
        status("识别底牌")
        val three = knownThree ?: run {
            grab()
            val h = cards('m', Regions.MY_HAND)
            (if (h.size == 20) threeFromHand(h.env()) else null) ?: waitThree()
        }
        listener.onThreeCards(Cards.formatDesc(three))
        sleep(500)
        status("识别角色")
        var pos: Position? = null
        while (pos == null) {
            grab()
            checkMeta()
            pos = findRole()
            if (pos == null) sleep(200)
        }
        listener.onRole(pos)
        val hand = waitHand(if (pos == Position.LANDLORD) 20 else 17)
        val state = GameState(pos, hand.env(), three)
        playedThisRun = true
        log("我的角色：${pos.label}，手牌：${hand.sortedText()}，底牌：${Cards.formatDesc(three)}")
        listener.onHand(Cards.formatDesc(state.hand))
        updateRemaining(state)
        status("对局中")

        // 0 = me, 1 = right (下家), 2 = left (上家)
        var order = when (pos) {
            Position.LANDLORD -> 0
            Position.LANDLORD_UP -> 1
            Position.LANDLORD_DOWN -> 2
        }
        var first = true
        while (!state.isOver || !confirmGameOver(state)) {
            when (order) {
                0 -> {
                    myTurn(state, first)
                    first = false
                }
                1 -> otherTurn(state, Seat.RIGHT)
                else -> otherTurn(state, Seat.LEFT)
            }
            order = (order + 1) % 3
        }
        status("本局结束")
        sleep(3000)
    }

    /**
     * The tracked card counts say someone is out of cards. A misread move
     * (e.g. a long straight read with extra cards) can cause that while the
     * game goes on, so look at the table for a few seconds: if our buttons
     * show up or the next player moves, fix the counts and keep playing.
     */
    private fun confirmGameOver(state: GameState): Boolean {
        if (state.hand.isEmpty()) return true
        val seat = when (state.acting) {
            state.user -> Seat.ME
            state.user.next() -> Seat.RIGHT
            else -> Seat.LEFT
        }
        // an old move may stay on screen after the end: only a change counts
        grab()
        val oldCards = cards(seat.set, seat.played).sortedText()
        val oldPass = vision.locate("buchu", seat.pass) != null
        val deadline = now() + 5000
        while (now() < deadline) {
            grab()
            checkMeta()
            val goesOn = if (seat == Seat.ME) {
                listOf("play_card", "pass_btn", "yaobuqi").any { vision.locate(it, Regions.BUTTONS) != null }
            } else {
                val cur = cards(seat.set, seat.played).sortedText()
                (cur.isNotEmpty() && cur != oldCards) || (!oldPass && vision.locate("buchu", seat.pass) != null)
            }
            if (goesOn) {
                val unseen = state.otherHandCards().size
                val others = Position.entries.filter { it != state.user }
                for (p in others) {
                    if (state.cardsLeft.getValue(p) > 0) continue
                    val other = others.first { it != p }
                    state.cardsLeft[p] = maxOf(1, unseen - state.cardsLeft.getValue(other))
                    log("警告：${p.label}的剩余牌数被识别成 0，但对局还在继续，已改为 ${state.cardsLeft[p]} 张")
                }
                return false
            }
            sleep(200)
        }
        return true
    }

    private fun record(state: GameState, move: List<Int>) {
        if (state.acting != state.user && move.size > state.cardsLeft.getValue(state.acting)) {
            log("警告：${state.acting.label}出了 ${move.size} 张，但记录中只剩 ${state.cardsLeft.getValue(state.acting)} 张")
        }
        if (state.acting != state.user && move.isNotEmpty()) {
            val unseen = state.otherHandCards().toMutableList()
            if (!move.all { unseen.remove(it) }) log("警告：识别到的出牌 ${Cards.formatDesc(move)} 与记牌器不符，可能识别有误")
        }
        state.play(move)
        listener.onHand(Cards.formatDesc(state.hand))
        updateRemaining(state)
    }

    private fun updateRemaining(state: GameState) {
        val unseen = Cards.formatDesc(state.otherHandCards())
        listener.onRemaining(IntArray(Cards.REAL_DESC.length) { i -> unseen.count { it == Cards.REAL_DESC[i] } })
    }

    private fun myTurn(state: GameState, first: Boolean) {
        waitMyTurnStart()
        var shown = cards('c', Regions.MY_PLAYED)
        if (shown.isNotEmpty()) {
            sleep(500)
            grab()
            shown = cards('c', Regions.MY_PLAYED)
        }
        if (shown.isNotEmpty() && isPlayable(state, shown.env())) {
            log("已经出过牌：${shown.sortedText()}")
            record(state, shown.env())
            return
        }

        val d = ai.agent.decide(state)
        listener.onSuggestion(if (d.action.isEmpty()) "不出" else Cards.formatDesc(d.action))
        listener.onCandidates(d.topText())
        if (d.ranked.isNotEmpty()) listener.onBidScore("当前得分: ${fmt(d.ranked[0].value * 8.0)}")
        d.note?.let { log("【$it】") }

        if (!settings.autoPlay) {
            status("轮到你出牌，推荐：${if (d.action.isEmpty()) "不出" else Cards.formatDesc(d.action)}")
            val move = waitMyMove(state)
            record(state, move)
            status("对局中")
            return
        }

        if (settings.thinkDelayMs > 0) sleep(settings.thinkDelayMs)
        val played = if (d.action.isEmpty()) {
            doPass()
            emptyList()
        } else if (doPlay(d.action, first)) {
            d.action
        } else {
            log("自动出牌未成功，改为读取实际出牌")
            waitMyMove(state)
        }
        record(state, played)
        log("出牌：${if (played.isEmpty()) "Pass" else Cards.formatDesc(played)} | 得分：${fmt(d.winRate)} | 剩余手牌：${Cards.formatDesc(state.hand)}")
        if (played.isNotEmpty() && isAnimated(played)) sleep(500)
    }

    private fun otherTurn(state: GameState, seat: Seat) {
        val text = "等待${seat.label}出牌"
        if (seat == Seat.RIGHT) listener.onRightPlayed(text) else listener.onLeftPlayed(text)
        waitTurnOver(seat)
        // the landlord's first move is never a pass (old 不加倍 labels may still be on screen)
        val move = waitMove(seat, allowPass = state.history.isNotEmpty())
        val who = state.acting
        record(state, move)
        val shown = if (move.isEmpty()) "不出" else Cards.formatDesc(move)
        if (seat == Seat.RIGHT) listener.onRightPlayed(shown) else listener.onLeftPlayed(shown)
        log("${seat.label}（${who.label}）出牌：$shown")
    }

    /**
     * Wait while [seat] is thinking: the game shows the alarm clock next to
     * that player. Until the game clears it, the area still shows the
     * player's move (or 不出) from the previous round, which must not be
     * read as the new one. If the clock never shows up (missed or not
     * recognised) fall back to reading after a short delay.
     */
    private fun waitTurnOver(seat: Seat) {
        val start = now()
        var sawClock = false
        while (true) {
            grab()
            checkMeta()
            if (vision.locate(seat.clock, seat.clockRegion) != null) {
                sawClock = true
            } else if (sawClock || now() - start > CLOCK_FALLBACK_MS) {
                return
            }
            sleep(100)
        }
    }

    /** Our turn has started once our clock or buttons show up; until then our old move may still be shown. */
    private fun waitMyTurnStart() {
        val start = now()
        while (true) {
            grab()
            checkMeta()
            if (vision.locate(Seat.ME.clock, Seat.ME.clockRegion) != null) return
            if (MY_BUTTONS.any { vision.locate(it, Regions.BUTTONS) != null }) return
            if (now() - start > CLOCK_FALLBACK_MS) return
            sleep(100)
        }
    }

    /** Wait for [seat]'s move to appear and read it until stable (port of the play_order loops). */
    private fun waitMove(seat: Seat, allowPass: Boolean = true): List<Int> {
        while (true) {
            grab()
            checkMeta()
            val passed = allowPass && vision.locate("buchu", seat.pass) != null
            val row = cards(seat.set, seat.played)
            if (row.isNotEmpty()) break
            if (passed) return emptyList()
            if (hasAnimation()) sleep(200) else sleep(100)
        }
        var invalidReads = 0
        while (true) {
            grab()
            checkMeta()
            val one = cards(seat.set, seat.played).sortedText()
            sleep(300)
            grab()
            checkMeta()
            val two = cards(seat.set, seat.played).sortedText()
            if (one == two && one.isNotEmpty()) {
                var result = one
                if (('X' in one || 'D' in one) && !("DX" in one)) {
                    sleep(500)
                    grab()
                    val again = cards(seat.set, seat.played).sortedText()
                    if (again.isNotEmpty()) result = again
                }
                // long moves (straights, planes...) fan out with an animation: read again once it settled
                if (result.length >= 5) {
                    sleep(700)
                    grab()
                    val a = cards(seat.set, seat.played).sortedText()
                    sleep(300)
                    grab()
                    val b = cards(seat.set, seat.played).sortedText()
                    if (a == b && a.length > result.length && moveType(Cards.parse(a)).type != MoveTypes.WRONG) {
                        log("${seat.label}的长出牌重读：$result -> $a")
                        result = a
                    }
                }
                val move = Cards.parse(result)
                if (moveType(move).type != MoveTypes.WRONG) return move
                if (++invalidReads >= 5) {
                    log("警告：${seat.label}出牌 $result 不是合法牌型，可能识别有误")
                    return move
                }
            } else if (allowPass && two.isEmpty() && vision.locate("buchu", seat.pass) != null) {
                return emptyList()
            }
        }
    }

    /** [move] can be our next move: we hold the cards and it is legal against the table. */
    private fun isPlayable(state: GameState, move: List<Int>): Boolean {
        val sorted = move.sorted()
        return state.legalActions().any { it.sorted() == sorted }
    }

    /**
     * Our own move when we did not play it automatically (hint mode or failed
     * taps): read it from our played-cards area like the others' moves, or,
     * once the play buttons are gone, from the cards that left the hand.
     */
    private fun waitMyMove(state: GameState): List<Int> {
        var sawButtons = false
        var goneSince = 0L
        while (true) {
            grab()
            checkMeta()
            val shown = cards('c', Regions.MY_PLAYED)
            if (shown.isNotEmpty() && isPlayable(state, shown.env())) return waitMove(Seat.ME)
            if (vision.locate("buchu", Regions.MY_PASS) != null) return emptyList()
            val buttons = vision.locate("play_card", Regions.BUTTONS, 0.7) != null ||
                vision.locate("pass_btn", Regions.BUTTONS, 0.7) != null ||
                vision.locate("yaobuqi", Regions.BUTTONS) != null
            if (buttons) {
                sawButtons = true
                goneSince = 0L
            } else {
                if (goneSince == 0L) goneSince = now()
                if (now() - goneSince >= 800) {
                    // a pass leaves the hand unchanged: only trust it after we saw our buttons
                    handDiff(state)?.let { if (it.isNotEmpty() || sawButtons) return it }
                }
            }
            sleep(150)
        }
    }

    /** Cards that left the hand since the last recorded move, when the hand reads the same twice and that is a legal move. */
    private fun handDiff(state: GameState): List<Int>? {
        val one = cards('m', Regions.MY_HAND).env().sorted()
        sleep(300)
        grab()
        val two = cards('m', Regions.MY_HAND).env().sorted()
        if (one != two) return null
        val rest = state.hand.toMutableList()
        if (!one.all { rest.remove(it) }) return null
        return if (isPlayable(state, rest)) rest.sorted() else null
    }

    private fun hasAnimation(): Boolean {
        val regions = listOf(Regions.ANI_RIGHT, Regions.ANI_LEFT, Regions.ANI_ME)
        var last = regions.map { vision.snapshot(it) }
        try {
            repeat(2) {
                sleep(100)
                grab()
                val cur = regions.map { vision.snapshot(it) }
                val changed = last.indices.any { Vision.difference(last[it], cur[it]) > 2.0 }
                last.forEach { it.release() }
                last = cur
                if (changed) return true
            }
            return false
        } finally {
            last.forEach { it.release() }
        }
    }

    private fun isAnimated(move: List<Int>): Boolean {
        val t = moveType(move).type
        return t in setOf(4, 5, 13, 14, 8, 9, 10, 11, 12) || move.size >= 6
    }

    private fun tapCard(hit: CardHit) = tapWork(hit.match.cx, hit.match.cy + settings.tapOffsetY)

    private fun minus(a: List<Char>, b: List<Char>): List<Char> {
        val left = a.toMutableList()
        for (c in b) left.remove(c)
        return left
    }

    /**
     * Select exactly [target] (env cards) in the hand. Selected cards sit higher
     * than the others, so the selection is read back and corrected.
     */
    private fun selectCards(target: List<Int>): Boolean {
        val want = Cards.format(target).toList()
        var baseline: Int? = null
        for (attempt in 0 until 6) {
            grab()
            checkMeta()
            val row = cards('m', Regions.MY_HAND)
            if (row.isEmpty()) {
                sleep(300)
                continue
            }
            val base = baseline ?: row.hits.maxOf { it.match.y }.also { baseline = it }
            val selected = row.hits.filter { it.match.y < base - POP_THRESHOLD }
            val unselected = row.hits.filter { it.match.y >= base - POP_THRESHOLD }
            val need = minus(want, selected.map { it.rank })
            val extra = minus(selected.map { it.rank }, want)
            if (need.isEmpty() && extra.isEmpty()) return true
            if (attempt == 5) break

            val taps = ArrayList<CardHit>()
            for (c in extra) selected.lastOrNull { it.rank == c && it !in taps }?.let { taps += it }
            for (c in need) {
                val h = unselected.lastOrNull { it.rank == c && it !in taps }
                if (h == null) {
                    log("手牌中没找到 $c（手牌识别：${row.text}）")
                    return false
                }
                taps += h
            }
            // first attempt: tap everything; afterwards one card at a time
            val batch = if (attempt == 0) taps else taps.take(1)
            for (h in batch) {
                tapCard(h)
                sleep(120)
            }
            sleep(350)
        }
        return false
    }

    private fun waitButton(names: List<String>, timeoutMs: Long, alsoDone: (() -> Boolean)? = null): Match? {
        val deadline = now() + timeoutMs
        while (true) {
            grab()
            checkMeta()
            if (alsoDone?.invoke() == true) return null
            for (n in names) vision.locate(n, Regions.BUTTONS, 0.7)?.let { return it }
            if (now() > deadline) return null
            sleep(150)
        }
    }

    private fun doPlay(action: List<Int>, first: Boolean): Boolean {
        repeat(3) {
            if (!selectCards(action)) return@repeat
            val btn = waitButton(listOf("play_card"), 15000) ?: return@repeat
            if (first) sleep(500)
            tapMatch(btn)
            sleep(400)
            grab()
            if (vision.locate("play_card", Regions.BUTTONS, 0.7) == null) return true
            log("出牌按钮仍在，重新选牌")
        }
        return false
    }

    private fun doPass() {
        repeat(3) {
            val btn = waitButton(listOf("pass_btn", "yaobuqi"), 15000) {
                vision.locate("buchu", Regions.MY_PASS) != null
            } ?: return
            tapMatch(btn)
            sleep(400)
            grab()
            if (vision.locate("pass_btn", Regions.BUTTONS, 0.7) == null &&
                vision.locate("yaobuqi", Regions.BUTTONS, 0.7) == null
            ) return
        }
    }

    companion object {
        /** A selected card's glyph is at least this many work pixels above the others. */
        const val POP_THRESHOLD = 10

        /** Buttons of our own turn while playing. */
        val MY_BUTTONS = listOf("play_card", "pass_btn", "yaobuqi")

        /** How long to look for the next player's clock before reading their area anyway. */
        const val CLOCK_FALLBACK_MS = 3000L

        /** Buttons that only show up at a game table. */
        val TABLE_BUTTONS = listOf(
            "jiaodizhu_btn", "qiangdizhu_btn", "jiabei_btn", "bujiabei_btn", "play_card", "pass_btn",
        )

        val POPUPS = listOf(
            // after a game: give up instead of watching a video / using up 回归特权
            "fangqi" to Regions.FANGQI, // 返豆暴击
            "fangqi_win" to Regions.FANGQI, // 赢豆暴击
            "fangqi_star" to Regions.FANGQI, // 星星保护
            "quick_start" to Regions.QUICK_START,
            "reward" to Regions.REWARD,
            "start_game" to Regions.START_GAME,
            "tuoguan" to Regions.TUOGUAN,
            "sure" to Regions.SURE,
            "tuijian" to Regions.TUIJIAN,
            "good" to Regions.GOOD,
            "zhidao" to Regions.ZHIDAO,
            "chacha" to Regions.CHACHA,
        )
    }
}
