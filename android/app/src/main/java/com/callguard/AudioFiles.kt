package com.callguard

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.provider.DocumentsContract
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

object AudioFiles {
    private val AUDIO_EXT = Regex(".*\\.(m4a|aac|amr|3gp|mp3|wav|ogg|opus|flac)$", RegexOption.IGNORE_CASE)

    /** Decode any audio the phone supports (m4a, amr, mp3, wav, ogg, ...) -> 16 kHz mono floats. */
    fun decode(ctx: Context, uri: Uri, maxSeconds: Int = 120, targetSr: Int = 16000): FloatArray {
        val ex = MediaExtractor()
        ex.setDataSource(ctx, uri, null)
        val track = (0 until ex.trackCount).firstOrNull {
            ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: throw IllegalArgumentException("no audio track")
        ex.selectTrack(track)
        val fmt = ex.getTrackFormat(track)
        var sr = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var ch = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var floatPcm = false
        val codec = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(fmt, null, null, 0)
        codec.start()

        var out = FloatArray(sr * 10)
        var n = 0
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        try {
            while (true) {
                if (!inputDone) {
                    val i = codec.dequeueInputBuffer(10_000)
                    if (i >= 0) {
                        val size = ex.readSampleData(codec.getInputBuffer(i)!!, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(i, 0, size, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }
                val o = codec.dequeueOutputBuffer(info, 10_000)
                if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val f = codec.outputFormat
                    sr = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                    ch = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    floatPcm = f.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                        f.getInteger(MediaFormat.KEY_PCM_ENCODING) == AudioFormat.ENCODING_PCM_FLOAT
                } else if (o >= 0) {
                    val buf = codec.getOutputBuffer(o)!!.order(ByteOrder.nativeOrder())
                    buf.position(info.offset); buf.limit(info.offset + info.size)
                    val frames = if (floatPcm) info.size / 4 / ch else info.size / 2 / ch
                    if (n + frames > out.size) out = out.copyOf(maxOf(out.size * 2, n + frames))
                    if (floatPcm) {
                        val fb = buf.asFloatBuffer()
                        for (k in 0 until frames) { var s = 0f; for (c in 0 until ch) s += fb.get(); out[n++] = s / ch }
                    } else {
                        val sb = buf.asShortBuffer()
                        for (k in 0 until frames) { var s = 0f; for (c in 0 until ch) s += sb.get() / 32768f; out[n++] = s / ch }
                    }
                    codec.releaseOutputBuffer(o, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0 || n >= sr * maxSeconds) break
                }
            }
        } finally {
            codec.stop(); codec.release(); ex.release()
        }
        return resample(out.copyOf(n), sr, targetSr)
    }

    /** Linear-interpolation resampler. When downsampling (44.1/48 kHz voice notes) the input is
     *  low-passed first, like training's resample_poly; otherwise content above 8 kHz would alias
     *  into exactly the band where the model looks for synthesis artefacts. */
    fun resample(input: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || input.isEmpty()) return input
        val x = if (to < from) lowPass(input, 0.45 * to / from) else input
        val n = (x.size.toLong() * to / from).toInt()
        val r = from.toDouble() / to
        return FloatArray(n) {
            val pos = it * r
            val i = pos.toInt()
            val f = (pos - i).toFloat()
            if (i + 1 < x.size) x[i] * (1 - f) + x[i + 1] * f else x[x.size - 1]
        }
    }

    /** Hann-windowed sinc FIR; `cutoff` is a fraction of the input sample rate (0.5 = Nyquist). */
    private fun lowPass(x: FloatArray, cutoff: Double, taps: Int = 63): FloatArray {
        val h = FloatArray(taps)
        val m = taps / 2
        var sum = 0.0
        for (k in 0 until taps) {
            val t = (k - m).toDouble()
            val sinc = if (t == 0.0) 2 * cutoff else sin(2 * PI * cutoff * t) / (PI * t)
            val w = 0.5 - 0.5 * cos(2 * PI * k / (taps - 1))
            h[k] = (sinc * w).toFloat(); sum += h[k]
        }
        for (k in 0 until taps) h[k] = (h[k] / sum).toFloat()  // unity gain at DC
        return FloatArray(x.size) { i ->
            var acc = 0f
            for (k in maxOf(0, i + m - x.size + 1) until minOf(taps, i + m + 1)) acc += h[k] * x[i + m - k]
            acc
        }
    }

    /** Newest audio file in a folder the user picked (e.g. the dialer's call-recordings folder). */
    fun newestIn(ctx: Context, tree: Uri, modifiedAfterMs: Long = 0): Uri? {
        val children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        val cols = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_MIME_TYPE,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME, DocumentsContract.Document.COLUMN_LAST_MODIFIED,
        )
        var best: String? = null
        var bestTime = modifiedAfterMs
        ctx.contentResolver.query(children, cols, null, null, null)?.use { c ->
            while (c.moveToNext()) {
                val mime = c.getString(1) ?: ""
                val name = c.getString(2) ?: ""
                val t = c.getLong(3)
                if ((mime.startsWith("audio/") || AUDIO_EXT.matches(name)) && t >= bestTime) {
                    best = c.getString(0); bestTime = t
                }
            }
        }
        return best?.let { DocumentsContract.buildDocumentUriUsingTree(tree, it) }
    }
}
