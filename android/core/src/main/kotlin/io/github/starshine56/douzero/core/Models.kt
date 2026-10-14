package io.github.starshine56.douzero.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.nio.FloatBuffer

/** DouZero card-play value network: z [n,40,54] + x [n,15] -> n values. */
fun interface ValueModel {
    fun predict(z: FloatArray, x: FloatArray, n: Int): FloatArray
}

/** Hand-only network (bid / farmer score): x [n,60] -> n values. */
fun interface HandModel {
    fun predict(x: FloatArray, n: Int): FloatArray
}

/** The six networks exported by tools/export_onnx.py. */
class ModelSet(
    val landlord: ValueModel,
    val landlordUp: ValueModel,
    val landlordDown: ValueModel,
    val bid: HandModel,
    val farmer: HandModel,
    val farmerUp: HandModel,
) {
    fun value(position: Position): ValueModel = when (position) {
        Position.LANDLORD -> landlord
        Position.LANDLORD_UP -> landlordUp
        Position.LANDLORD_DOWN -> landlordDown
    }
}

/**
 * ONNX Runtime backed models. The ONNX Runtime classes come from
 * onnxruntime-android on the phone and from the desktop jar in unit tests.
 */
class OrtModels(private val load: (String) -> ByteArray, private val threads: Int = 4) : Closeable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessions = ArrayList<OrtSession>()

    @Synchronized
    private fun session(name: String): OrtSession {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        return env.createSession(load(name), opts).also { sessions += it }
    }

    private inner class Value(private val name: String) : ValueModel {
        private val s: OrtSession by lazy(LazyThreadSafetyMode.SYNCHRONIZED) { session(name) }
        override fun predict(z: FloatArray, x: FloatArray, n: Int): FloatArray {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(z), longArrayOf(n.toLong(), 40, 54)).use { zt ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(n.toLong(), 15)).use { xt ->
                    s.run(mapOf("z" to zt, "x" to xt)).use { r -> return flatten(r, n) }
                }
            }
        }
    }

    private inner class Hand(private val name: String) : HandModel {
        private val s: OrtSession by lazy(LazyThreadSafetyMode.SYNCHRONIZED) { session(name) }
        override fun predict(x: FloatArray, n: Int): FloatArray {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(n.toLong(), 60)).use { xt ->
                s.run(mapOf("x" to xt)).use { r -> return flatten(r, n) }
            }
        }
    }

    private fun flatten(r: OrtSession.Result, n: Int): FloatArray {
        val t = r.get(0) as OnnxTensor
        val buf = t.floatBuffer
        return FloatArray(n) { buf.get(it) }
    }

    val models: ModelSet = ModelSet(
        landlord = Value("resnet_landlord.onnx"),
        landlordUp = Value("resnet_landlord_up.onnx"),
        landlordDown = Value("resnet_landlord_down.onnx"),
        bid = Hand("bid.onnx"),
        farmer = Hand("farmer.onnx"),
        farmerUp = Hand("farmer_up.onnx"),
    )

    override fun close() {
        sessions.forEach { it.close() }
        sessions.clear()
    }

    companion object {
        val FILES = listOf(
            "resnet_landlord.onnx", "resnet_landlord_up.onnx", "resnet_landlord_down.onnx",
            "bid.onnx", "farmer.onnx", "farmer_up.onnx",
        )
    }
}
