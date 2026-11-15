package com.callguard

import android.Manifest
import android.accessibilityservice.AccessibilityService
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import kotlin.math.abs

/**
 * In-call checker. Android lets an accessibility service keep capturing the microphone while a
 * call is active (Android 10+ concurrent-capture rules), which ordinary apps and browsers can't.
 * The service reads no screen content; it only detects calls, draws a bubble and listens.
 */
class GuardService : AccessibilityService() {

    companion object {
        @Volatile var running: GuardService? = null
        private const val TAG = "CallGuard"
    }

    private val main = Handler(Looper.getMainLooper())
    private lateinit var wm: WindowManager
    private lateinit var audio: AudioManager
    private var bubble: TextView? = null
    private var alertView: View? = null
    @Volatile private var capturing = false
    /** Bumped on every start/stop; a capture thread runs only while its generation is current,
     *  so a quick stop -> start never leaves two threads (and two AudioRecords) running. */
    @Volatile private var captureGen = 0
    private var captureThread: Thread? = null
    private var inCall = false
    private var callStartMs = 0L
    private var blockedNotified = false
    private var modeListener: AudioManager.OnModeChangedListener? = null
    private val pollMode = object : Runnable {
        override fun run() { onMode(audio.mode); main.postDelayed(this, 1000) }
    }

    // ------------------------------------------------------------ lifecycle

    override fun onServiceConnected() {
        running = this
        wm = getSystemService(WindowManager::class.java)
        audio = getSystemService(AudioManager::class.java)
        Notify.channels(this)
        watchCalls()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}

    override fun onDestroy() {
        stopCapture()
        removeBubble(); dismissAlert()
        if (Build.VERSION.SDK_INT >= 31) modeListener?.let { audio.removeOnModeChangedListener(it) }
        main.removeCallbacks(pollMode)
        running = null
        super.onDestroy()
    }

    // ------------------------------------------------------------ call detection

    /** Audio mode IN_CALL (cellular) / IN_COMMUNICATION (WhatsApp & other VoIP) = a call is live.
     *  Android 12+ reports mode changes; on 10-11 the mode is polled once a second (cheap, and
     *  unlike the call-state listener it also sees VoIP calls). */
    private fun watchCalls() {
        if (Build.VERSION.SDK_INT >= 31) {
            val l = AudioManager.OnModeChangedListener { onMode(it) }
            audio.addOnModeChangedListener(mainExecutor, l)
            modeListener = l
            onMode(audio.mode)
        } else {
            main.post(pollMode)
        }
    }

    private fun onMode(mode: Int) =
        onCallActive(mode == AudioManager.MODE_IN_CALL || mode == AudioManager.MODE_IN_COMMUNICATION)

    private fun onCallActive(active: Boolean) {
        if (active && !inCall) {
            inCall = true
            callStartMs = System.currentTimeMillis()
            blockedNotified = false
            when (Prefs.mode(this)) {
                Mode.OFF -> {}
                Mode.ON_REQUEST -> showBubble("Check\ncaller", Color.DKGRAY)
                Mode.AUTO -> { showBubble("Starting…", Color.DKGRAY); startCapture() }
            }
        } else if (!active && inCall) {
            inCall = false
            stopCapture()
            removeBubble()
            afterCall()
        }
    }

    private fun afterCall() {
        val folder = Prefs.folder(this) ?: return
        if (Prefs.mode(this) == Mode.OFF) return
        // dialers finish writing the recording a moment after the call ends
        main.postDelayed({
            runCatching { AudioFiles.newestIn(this, folder, callStartMs - 60_000) }
                .onSuccess { it?.let { uri -> Notify.afterCall(this, uri) } }
                .onFailure { Log.w(TAG, "recordings folder: $it") }
        }, 4000)
    }

    // ------------------------------------------------------------ capture + scoring loop

    private fun startCapture() {
        if (capturing) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            updateBubble("Mic\npermission?", Color.DKGRAY); return
        }
        capturing = true
        val gen = ++captureGen
        val prev = captureThread
        captureThread = Thread({
            prev?.join()  // the old loop exits within one 100 ms read: never hold the mic twice
            captureLoop(gen)
        }, "callguard-capture").also { it.start() }
    }

    private fun stopCapture() {
        capturing = false
        captureGen++
    }

    @SuppressLint("MissingPermission")
    private fun openRecorder(sr: Int): AudioRecord? {
        val min = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        for (src in intArrayOf(MediaRecorder.AudioSource.VOICE_RECOGNITION, MediaRecorder.AudioSource.MIC)) {
            val r = runCatching {
                AudioRecord(src, sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(min, sr / 2))
            }.getOrNull() ?: continue
            if (r.state == AudioRecord.STATE_INITIALIZED) {
                r.startRecording()
                if (r.recordingState == AudioRecord.RECORDSTATE_RECORDING) return r
            }
            r.release()
        }
        return null
    }

    private fun captureLoop(gen: Int) {
        if (gen != captureGen) return
        // UI updates from a loop that has since been stopped (bubble tapped, call ended) are dropped
        fun ui(r: () -> Unit) = post { if (gen == captureGen) r() }
        val det = Detector.get(this)
        val rec = openRecorder(det.sr)
        if (rec == null) {
            ui { updateBubble("Mic\nblocked", Color.GRAY); notifyBlocked() }
            if (gen == captureGen) capturing = false
            return
        }
        ui { updateBubble(if (det.hasModel) "Listening…\n(speaker on)" else "Mic test", Color.DKGRAY) }
        val ring = FloatArray(det.window)
        val chunk = ShortArray(det.sr / 10)
        var filled = 0; var sinceHop = 0; var sinceMeter = 0; var streak = 0; var zeroHops = 0
        try {
            while (capturing && gen == captureGen) {
                val n = rec.read(chunk, 0, chunk.size)
                if (n <= 0) { if (n < 0) break else continue }
                System.arraycopy(ring, n, ring, 0, ring.size - n)
                for (i in 0 until n) ring[ring.size - n + i] = chunk[i] / 32768f
                filled += n; sinceHop += n; sinceMeter += n

                if (!det.hasModel && sinceMeter >= det.sr / 2) {  // diagnostic mode: live mic level
                    sinceMeter = 0
                    val last = ring.copyOfRange(ring.size - det.sr / 2, ring.size)
                    val db = Detector.dbfs(last)
                    val allZero = last.all { it == 0f }
                    ui { updateBubble(if (allZero) "Mic\nBLOCKED" else "Mic OK\n${db.toInt()} dB",
                        if (allZero) Color.GRAY else Color.rgb(37, 99, 235)) }
                    if (allZero && ++zeroHops >= 6) ui { notifyBlocked() }
                    if (!allZero) zeroHops = 0
                }
                if (!det.hasModel || filled < ring.size || sinceHop < det.hop) continue
                sinceHop = 0

                if (Detector.peak(ring) == 0f) {  // exact digital silence => the OS is muting us
                    if (++zeroHops >= 2) ui { updateBubble("Mic\nblocked", Color.GRAY); notifyBlocked() }
                    continue
                }
                zeroHops = 0
                if (!Detector.hasSpeech(ring)) { ui { updateBubble("Listening…", Color.DKGRAY) }; continue }

                val t0 = System.nanoTime()
                val p = det.score(ring.copyOf())
                val lvl = det.level(p, det.liveThreshold)
                streak = if (lvl == Level.RED) streak + 1 else 0
                Log.d(TAG, "p=%.3f level=%s streak=%d %.0f ms".format(p, lvl, streak, (System.nanoTime() - t0) / 1e6))
                val colour = when (lvl) {
                    Level.GREEN -> Color.rgb(22, 163, 74)
                    Level.YELLOW -> Color.rgb(202, 138, 4)
                    Level.RED -> Color.rgb(220, 38, 38)
                }
                val label = when (lvl) { Level.GREEN -> "Low risk"; Level.YELLOW -> "Some\nsigns"; Level.RED -> "AI voice\nrisk!" }
                ui { updateBubble("$label\n${(det.risk(p, det.liveThreshold) * 100).toInt()}%", colour) }
                if (streak == det.consecutive) ui { raiseAlert() }
            }
        } finally {
            rec.stop(); rec.release()
            ring.fill(0f)  // nothing is kept
        }
    }

    private fun notifyBlocked() {
        if (blockedNotified) return
        blockedNotified = true
        Notify.blocked(this)
    }

    // ------------------------------------------------------------ UI overlays (no extra permission)

    private fun dp(v: Int) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt()
    private fun post(r: () -> Unit) { main.post { r() } }

    private fun overlayParams(w: Int, h: Int) = WindowManager.LayoutParams(
        w, h, WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
        PixelFormat.TRANSLUCENT,
    )

    @SuppressLint("ClickableViewAccessibility")
    private fun showBubble(text: String, colour: Int) {
        if (bubble != null) { updateBubble(text, colour); return }
        val v = TextView(this).apply {
            gravity = Gravity.CENTER
            setTextColor(Color.WHITE)
            textSize = 12f
            setPadding(dp(6), dp(6), dp(6), dp(6))
            background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(colour) }
            this.text = text
            contentDescription = "Call Guard"
        }
        val lp = overlayParams(dp(84), dp(84)).apply {
            gravity = Gravity.TOP or Gravity.END; x = dp(12); y = dp(160)
        }
        var downX = 0f; var downY = 0f; var startX = 0; var startY = 0
        v.setOnTouchListener { _, e ->
            when (e.action) {
                MotionEvent.ACTION_DOWN -> { downX = e.rawX; downY = e.rawY; startX = lp.x; startY = lp.y }
                MotionEvent.ACTION_MOVE -> {
                    lp.x = startX - (e.rawX - downX).toInt(); lp.y = startY + (e.rawY - downY).toInt()
                    runCatching { wm.updateViewLayout(v, lp) }
                }
                MotionEvent.ACTION_UP -> if (abs(e.rawX - downX) < dp(8) && abs(e.rawY - downY) < dp(8)) onBubbleTap()
            }
            true
        }
        runCatching { wm.addView(v, lp); bubble = v }.onFailure { Log.w(TAG, "bubble: $it") }
    }

    private fun onBubbleTap() {
        if (capturing) {
            stopCapture()
            updateBubble("Check\ncaller", Color.DKGRAY)
        } else {
            updateBubble("Turn on\nspeaker", Color.DKGRAY)
            startCapture()
        }
    }

    private fun updateBubble(text: String, colour: Int) {
        bubble?.apply { this.text = text; (background as GradientDrawable).setColor(colour) }
    }

    private fun removeBubble() {
        bubble?.let { runCatching { wm.removeView(it) } }
        bubble = null
    }

    private fun raiseAlert() {
        val vib = getSystemService(Vibrator::class.java)
        vib?.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 400, 150, 400, 150, 400), -1))
        Notify.fakeVoice(this)
        if (alertView != null) return
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
            background = GradientDrawable().apply { cornerRadius = dp(16).toFloat(); setColor(Color.WHITE) }
            addView(TextView(context).apply {
                text = "⚠ This voice may be AI-generated"; textSize = 20f; setTextColor(Color.rgb(185, 28, 28))
            })
            addView(TextView(context).apply {
                textSize = 15f; setTextColor(Color.BLACK); setPadding(0, dp(10), 0, dp(10))
                text = "Verify before sending money or sharing an OTP.\n\n" +
                    "• Hang up and call back on their saved number.\n" +
                    "• Ask your family code word or something only they know.\n" +
                    "• Never share OTPs, PINs or passwords on a call."
            })
            addView(Button(context).apply { text = "I understand"; setOnClickListener { dismissAlert() } })
        }
        val frame = LinearLayout(this).apply {
            setPadding(dp(8), dp(8), dp(8), dp(8))
            setBackgroundColor(Color.rgb(185, 28, 28))
            addView(box, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        }
        // A banner at the top, not a full-screen dialog: the dialer's hang-up button (the advice
        // above) must stay reachable. Not focusable = touches outside the banner pass through.
        val lp = overlayParams(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.WRAP_CONTENT)
            .apply { gravity = Gravity.TOP }
        runCatching { wm.addView(frame, lp); alertView = frame }
    }

    private fun dismissAlert() {
        alertView?.let { runCatching { wm.removeView(it) } }
        alertView = null
    }
}
