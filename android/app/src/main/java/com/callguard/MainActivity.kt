package com.callguard

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView

/** Setup + settings + "check a recording". Plain framework views: no AndroidX, tiny APK. */
class MainActivity : Activity() {

    private val PICK_FILE = 1
    private val PICK_FOLDER = 2
    private lateinit var status: TextView
    private lateinit var result: TextView
    private lateinit var folderLabel: TextView
    private lateinit var col: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Notify.channels(this)
        col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(18), dp(24), dp(18), dp(24)) }
        setContentView(ScrollView(this).apply { setBackgroundColor(Color.rgb(15, 23, 42)); addView(col) })

        text("Call Guard", 26f, bold = true)
        text("Warns you if a caller's voice may be AI-generated. Everything runs on this phone; no audio is stored or sent.", 14f, muted = true)

        header("1 · Setup")
        status = text("", 15f)
        button("Allow microphone") { requestPerms() }
        text("Next, open Accessibility → \"Call Guard\" and switch it on. It only uses the microphone " +
            "during calls; it cannot read your screen.", 13f, muted = true)
        button("Turn on Call Guard service") { startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }

        header("2 · During calls")
        val modes = RadioGroup(this)
        for ((m, label) in listOf(
            Mode.OFF to "Off",
            Mode.ON_REQUEST to "On request – a \"Check caller\" bubble appears during calls; tap it to start",
            Mode.AUTO to "Automatic – start checking as soon as a call is answered",
        )) {
            modes.addView(RadioButton(this).apply {
                id = View.generateViewId(); text = label; setTextColor(Color.WHITE); tag = m
                isChecked = Prefs.mode(this@MainActivity) == m
                setOnCheckedChangeListener { _, on -> if (on) Prefs.setMode(this@MainActivity, m) }
            })
        }
        col.addView(modes)
        text("Put the call on SPEAKER so the phone's mic can hear the caller.", 13f, muted = true)

        header("3 · Check a recording or voice note")
        button("Choose audio file") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE); type = "audio/*"
            }, PICK_FILE)
        }
        folderLabel = text("", 13f, muted = true)
        button("Choose call-recordings folder") { startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), PICK_FOLDER) }
        button("Check latest call recording") {
            val f = Prefs.folder(this)
            val uri = f?.let { runCatching { AudioFiles.newestIn(this, it) }.getOrNull() }
            if (uri == null) show("No recording found. Choose your dialer's recordings folder first.", null) else check(uri)
        }
        text("Tip: you can also Share any voice note (e.g. from WhatsApp) to Call Guard.", 13f, muted = true)
        result = text("", 17f, bold = true)

        header("If a warning appears")
        text("• Hang up and call the person back on their saved number.\n• Ask your family code word.\n• Never share an OTP, PIN or password on a call.", 14f)

        handle(intent)
    }

    override fun onResume() {
        super.onResume()
        refreshStatus()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handle(intent)
    }

    private fun handle(i: Intent?) {
        i ?: return
        when {
            i.action == Intent.ACTION_SEND -> {
                @Suppress("DEPRECATION")
                (i.getParcelableExtra<Uri>(Intent.EXTRA_STREAM))?.let { check(it) }
            }
            i.action == Intent.ACTION_VIEW && i.data != null -> check(i.data!!)
            i.hasExtra(Notify.EXTRA_CHECK_URI) -> check(Uri.parse(i.getStringExtra(Notify.EXTRA_CHECK_URI)))
            i.getBooleanExtra(Notify.EXTRA_BLOCKED_HELP, false) -> show(
                "Your phone doesn't let apps hear calls live.\n\nOptions:\n• Turn on call recording in your dialer, " +
                    "choose its folder above, and Call Guard will offer to check each call after it ends.\n" +
                    "• Share a voice note to Call Guard.\n• Open the Call Guard web page on a 2nd phone/laptop near the speaker.",
                null,
            )
        }
    }

    // ------------------------------------------------------------ checking a file

    private fun check(uri: Uri) {
        val det = Detector.get(this)
        if (!det.hasModel) { show("No model in this build (diagnostic mode). Run scripts/export_onnx.py and rebuild.", null); return }
        show("Analysing…", null)
        Thread {
            val msg = runCatching {
                val x = AudioFiles.decode(this, uri)
                if (x.size < det.sr) error("clip is shorter than 1 second")
                val r = det.analyse(x)
                x.fill(0f)
                val fake = r.isFake(det.threshold)
                val pct = (det.risk(r.fakeProb) * 100).toInt()
                (if (fake) "⚠ Likely AI-generated (AI-voice risk $pct%)\nVerify before sending money or sharing an OTP."
                else "✓ Likely a real voice (AI-voice risk $pct%)\nNot a guarantee — stay careful with money requests.") +
                    "\n\n%.1f s analysed in %d window(s).".format(r.seconds, r.windowProbs.size) to fake
            }.getOrElse { ("Couldn't read this audio: ${it.message}" to null) }
            runOnUiThread { show(msg.first, msg.second) }
        }.start()
    }

    private fun show(msg: String, fake: Boolean?) {
        result.text = msg
        result.setTextColor(when (fake) { true -> Color.rgb(248, 113, 113); false -> Color.rgb(74, 222, 128); null -> Color.WHITE })
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data ?: return
        if (resultCode != RESULT_OK) return
        if (requestCode == PICK_FILE) check(uri)
        if (requestCode == PICK_FOLDER) {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            Prefs.setFolder(this, uri)
            refreshStatus()
        }
    }

    // ------------------------------------------------------------ permissions / status

    private fun requestPerms() {
        val perms = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= 33) perms += Manifest.permission.POST_NOTIFICATIONS
        requestPermissions(perms.toTypedArray(), 0)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        refreshStatus()
    }

    private fun serviceEnabled(): Boolean {
        if (GuardService.running != null) return true
        val list = Settings.Secure.getString(contentResolver, Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES) ?: return false
        return list.split(':').any { it.startsWith("$packageName/") }
    }

    private fun refreshStatus() {
        val mic = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        val det = Detector.get(this)
        fun mark(ok: Boolean) = if (ok) "✓" else "✗"
        status.text = "${mark(mic)} Microphone permission\n${mark(serviceEnabled())} Call Guard service\n" +
            if (det.hasModel) "✓ AI-voice model loaded (on-device)"
            else "• Diagnostic build: no model yet — the bubble shows the mic level so you can test whether your phone allows listening during calls"
        folderLabel.text = "Recordings folder: " + (Prefs.folder(this)?.lastPathSegment ?: "not set")
    }

    // ------------------------------------------------------------ tiny view helpers

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun text(s: String, size: Float, bold: Boolean = false, muted: Boolean = false) = TextView(this).apply {
        text = s; textSize = size
        setTextColor(if (muted) Color.rgb(148, 163, 184) else Color.WHITE)
        if (bold) setTypeface(typeface, Typeface.BOLD)
        setPadding(0, dp(4), 0, dp(4))
        col.addView(this)
    }

    private fun header(s: String) = text(s, 18f, bold = true).apply { setPadding(0, dp(20), 0, dp(6)) }

    private fun button(label: String, onClick: () -> Unit) = Button(this).apply {
        text = label; isAllCaps = false
        setOnClickListener { onClick() }
        col.addView(this)
    }
}
