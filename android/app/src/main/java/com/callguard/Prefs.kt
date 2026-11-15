package com.callguard

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.net.Uri

enum class Mode { OFF, ON_REQUEST, AUTO }

object Prefs {
    private fun sp(ctx: Context) = ctx.getSharedPreferences("callguard", Context.MODE_PRIVATE)

    fun mode(ctx: Context) = Mode.valueOf(sp(ctx).getString("mode", Mode.OFF.name)!!)
    fun setMode(ctx: Context, m: Mode) = sp(ctx).edit().putString("mode", m.name).apply()

    fun folder(ctx: Context) = sp(ctx).getString("folder", null)?.let(Uri::parse)
    fun setFolder(ctx: Context, u: Uri?) = sp(ctx).edit().putString("folder", u?.toString()).apply()
}

object Notify {
    private const val ALERTS = "alerts"
    private const val INFO = "info"
    const val EXTRA_CHECK_URI = "check_uri"
    const val EXTRA_BLOCKED_HELP = "blocked_help"

    fun channels(ctx: Context) {
        val nm = ctx.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(ALERTS, "AI-voice warnings", NotificationManager.IMPORTANCE_HIGH))
        nm.createNotificationChannel(NotificationChannel(INFO, "Call Guard info", NotificationManager.IMPORTANCE_DEFAULT))
    }

    private fun open(ctx: Context, code: Int, fill: Intent.() -> Unit) = PendingIntent.getActivity(
        ctx, code, Intent(ctx, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
            fill()
        }, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
    )

    private fun post(ctx: Context, id: Int, channel: String, title: String, text: String, pi: PendingIntent) {
        channels(ctx)
        val n = Notification.Builder(ctx, channel)
            .setSmallIcon(R.drawable.ic_shield).setContentTitle(title).setContentText(text)
            .setStyle(Notification.BigTextStyle().bigText(text))
            .setContentIntent(pi).setAutoCancel(true).build()
        runCatching { ctx.getSystemService(NotificationManager::class.java).notify(id, n) }
    }

    fun fakeVoice(ctx: Context) = post(
        ctx, 1, ALERTS, "⚠ This voice may be AI-generated",
        "Verify before sending money or sharing an OTP. Hang up and call back on their saved number, or ask your family code word.",
        open(ctx, 1) {},
    )

    fun blocked(ctx: Context) = post(
        ctx, 2, INFO, "Live checking isn't allowed on this phone",
        "Your phone blocks the microphone during calls. Use Call Guard on a 2nd device, or check the call recording / a voice note after the call.",
        open(ctx, 2) { putExtra(EXTRA_BLOCKED_HELP, true) },
    )

    fun afterCall(ctx: Context, recording: Uri) = post(
        ctx, 3, INFO, "Check this call's recording?",
        "Tap to check the caller's voice for AI generation (on this phone).",
        open(ctx, 3) { putExtra(EXTRA_CHECK_URI, recording.toString()) },
    )
}
