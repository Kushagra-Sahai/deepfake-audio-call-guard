package com.callguard

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max

enum class Level { GREEN, YELLOW, RED }

/** On-device scorer for 4 s / 16 kHz windows. Same maths as web/app.js and dfad/infer.py. */
class Detector private constructor(private val session: OrtSession?, cfg: JSONObject) {
    val sr = cfg.optInt("sr", 16000)
    val window = cfg.optInt("window", 64000)
    val hop = cfg.optInt("hop", 32000)
    val threshold = cfg.optDouble("threshold", 0.5).toFloat()
    /** For in-call listening: the caller comes through the loudspeaker, which raises every score,
     *  so this is calibrated on speakerphone audio (scripts/export_onnx.py). */
    val liveThreshold = cfg.optDouble("live_threshold", threshold.toDouble()).toFloat()
    val consecutive = cfg.optInt("consecutive", 3)

    /** false = no model bundled yet: the app runs in diagnostic mode (mic level only). */
    val hasModel get() = session != null

    /** Display scale shared with web/app.js and dfad/infer.py: the threshold maps to 0.5,
     *  so "risk >= 50%" always means the FAKE verdict. */
    fun risk(p: Float, t: Float = threshold) = if (p < t) 0.5f * p / t else 0.5f + 0.5f * (p - t) / (1 - t)

    fun level(p: Float, t: Float = threshold) = risk(p, t).let { r ->
        when {
            r >= 0.5f -> Level.RED
            r >= 0.35f -> Level.YELLOW
            else -> Level.GREEN
        }
    }

    @Synchronized
    fun score(win: FloatArray): Float =
        OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), FloatBuffer.wrap(win), longArrayOf(1, win.size.toLong()))
            .use { t -> session!!.run(mapOf("audio" to t)).use { r -> (r[0].value as FloatArray)[0] } }

    data class Result(val fakeProb: Float, val windowProbs: List<Float>, val seconds: Float) {
        fun isFake(threshold: Float) = fakeProb >= threshold
    }

    fun analyse(x: FloatArray): Result {
        val probs = windows(x).map { score(it) }
        return Result(probs.average().toFloat(), probs, x.size / sr.toFloat())
    }

    /** 4 s windows every 2 s; long clips get MAX_WINDOWS spread over the whole clip (as in
     *  dfad/data.py); short clips are repeat-padded (as in training). */
    fun windows(x: FloatArray): List<FloatArray> {
        if (x.size <= window) return listOf(FloatArray(window) { if (x.isEmpty()) 0f else x[it % x.size] })
        var starts = (0..x.size - window step hop).toList()
        if (starts.size > MAX_WINDOWS) {
            starts = List(MAX_WINDOWS) { (it.toLong() * (x.size - window) / (MAX_WINDOWS - 1)).toInt() }
        }
        return starts.map { x.copyOfRange(it, it + window) }
    }

    companion object {
        private const val MAX_WINDOWS = 30
        @Volatile private var instance: Detector? = null

        fun get(ctx: Context): Detector = instance ?: synchronized(this) {
            instance ?: load(ctx.applicationContext).also { instance = it }
        }

        private fun load(ctx: Context): Detector {
            val cfg = runCatching {
                JSONObject(ctx.assets.open("config.json").bufferedReader().readText())
            }.getOrDefault(JSONObject())
            val session = runCatching {
                val bytes = ctx.assets.open("student.onnx").readBytes()
                val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
                OrtEnvironment.getEnvironment().createSession(bytes, opts)
            }.onFailure { Log.w("CallGuard", "no model bundled, diagnostic mode: $it") }.getOrNull()
            return Detector(session, cfg)
        }

        /** Enough 20 ms frames above -45 dBFS to be worth scoring. */
        fun hasSpeech(x: FloatArray): Boolean {
            val f = 320
            var voiced = 0
            var frames = 0
            var i = 0
            while (i + f <= x.size) {
                var e = 0.0
                for (j in i until i + f) e += x[j] * x[j]
                if (10 * log10(e / f + 1e-12) > -45) voiced++
                frames++; i += f
            }
            return frames > 0 && voiced.toFloat() / frames > 0.25f
        }

        fun peak(x: FloatArray): Float {
            var m = 0f
            for (v in x) m = max(m, abs(v))
            return m
        }

        fun dbfs(x: FloatArray): Float {
            var e = 0.0
            for (v in x) e += v * v
            return (10 * log10(e / max(1, x.size) + 1e-12)).toFloat()
        }
    }
}
