"""Builds docs/CallGuard_Report.pdf (ReportLab, vector flowcharts)."""
import re
import sys
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String, PolyLine
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, CondPageBreak)

PROJ = Path(r"C:\Users\HP\OneDrive\Desktop\audio")
OUT = PROJ / "docs" / "CallGuard_Report.pdf"
F = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("UI", f"{F}\\segoeui.ttf"))
pdfmetrics.registerFont(TTFont("UIB", f"{F}\\segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("UII", f"{F}\\segoeuii.ttf"))
pdfmetrics.registerFont(TTFont("UISB", f"{F}\\seguisb.ttf"))
pdfmetrics.registerFont(TTFont("Mono", f"{F}\\consola.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("UI", normal="UI", bold="UIB", italic="UII", boldItalic="UIB")

# ---------------------------------------------------------------- palette
NAVY = colors.HexColor("#0f172a")
INK = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
LINE = colors.HexColor("#cbd5e1")
BLUE = colors.HexColor("#2563eb")
BLUE_BG = colors.HexColor("#eff6ff")
GREEN = colors.HexColor("#16a34a")
GREEN_BG = colors.HexColor("#f0fdf4")
AMBER = colors.HexColor("#b45309")
AMBER_BG = colors.HexColor("#fffbeb")
RED = colors.HexColor("#dc2626")
RED_BG = colors.HexColor("#fef2f2")
VIOLET = colors.HexColor("#7c3aed")
VIOLET_BG = colors.HexColor("#f5f3ff")
GRAY_BG = colors.HexColor("#f1f5f9")
NAVY_BOX = colors.HexColor("#e2e8f0")

W, H = A4
M = 18 * mm
CW = W - 2 * M  # content width

# ---------------------------------------------------------------- text styles
ss = {
    "title": ParagraphStyle("title", fontName="UIB", fontSize=26, leading=32, textColor=NAVY),
    "subtitle": ParagraphStyle("subtitle", fontName="UI", fontSize=13, leading=18, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="UIB", fontSize=17, leading=22, textColor=NAVY, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="UISB", fontSize=12.5, leading=17, textColor=BLUE, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="UI", fontSize=10, leading=14.5, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="UI", fontSize=8.5, leading=11.5, textColor=MUTED, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="UI", fontSize=8.8, leading=11.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="UISB", fontSize=8.8, leading=11.5, textColor=INK),
    "head": ParagraphStyle("head", fontName="UISB", fontSize=8.8, leading=11.5, textColor=colors.white),
    "bullet": ParagraphStyle("bullet", fontName="UI", fontSize=10, leading=14.5, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=2),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.3, leading=11, textColor=INK, backColor=GRAY_BG,
                           borderPadding=(5, 6, 5, 6), spaceBefore=4, spaceAfter=8),
    "caption": ParagraphStyle("caption", fontName="UII", fontSize=8.5, leading=11, textColor=MUTED, alignment=TA_CENTER,
                              spaceBefore=2, spaceAfter=10),
    "callout": ParagraphStyle("callout", fontName="UI", fontSize=9.5, leading=13.5, textColor=INK),
}


def P(t, s="body"):
    return Paragraph(t, ss[s])


def bullets(items, s="bullet"):
    return [Paragraph(t, ss[s], bulletText="•") for t in items]


def code(t):
    return Paragraph(t.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"),
                     ss["code"])


def table(rows, widths, head=True, zebra=True, bold_first=False, align_right_from=None):
    data = []
    for i, r in enumerate(rows):
        st = "head" if (head and i == 0) else ("cellb" if bold_first else "cell")
        data.append([c if not isinstance(c, str) else Paragraph(c, ss["cellb" if (bold_first and j == 0 and i > 0)
                                                                     else ("head" if head and i == 0 else "cell")])
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 6),
             ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE)]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    if zebra:
        for i in range(1 if head else 0, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), GRAY_BG))
    t.setStyle(TableStyle(style))
    return t


def callout(text, fg=BLUE, bg=BLUE_BG, title=None):
    body = (f"<font name='UISB' color='{fg.hexval().replace('0x', '#')}'>{title}</font><br/>" if title else "") + text
    t = Table([[Paragraph(body, ss["callout"])]], colWidths=[CW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, fg),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


# ---------------------------------------------------------------- flowchart helpers
class FC:
    """Tiny flowchart canvas on a ReportLab Drawing (y grows upwards)."""

    def __init__(self, w, h):
        self.d = Drawing(w, h)
        self.w, self.h = w, h

    def _lines(self, text, size, font, x, y, color):
        lines = text.split("\n")
        lead = size * 1.25
        y0 = y + (len(lines) - 1) * lead / 2 - size * 0.35
        for i, ln in enumerate(lines):
            f = font
            if ln.startswith("**"):
                ln, f = ln[2:], "UIB"
            self.d.add(String(x, y0 - i * lead, ln, fontName=f, fontSize=size, fillColor=color, textAnchor="middle"))

    def box(self, cx, cy, w, h, text, fill=BLUE_BG, stroke=BLUE, size=8, font="UI", color=INK, r=5, dash=None):
        rect = Rect(cx - w / 2, cy - h / 2, w, h, rx=r, ry=r, fillColor=fill, strokeColor=stroke, strokeWidth=1)
        if dash:
            rect.strokeDashArray = dash
        self.d.add(rect)
        self._lines(text, size, font, cx, cy, color)
        return (cx, cy, w, h)

    def diamond(self, cx, cy, w, h, text, fill=AMBER_BG, stroke=AMBER, size=7.6):
        self.d.add(Polygon([cx, cy + h / 2, cx + w / 2, cy, cx, cy - h / 2, cx - w / 2, cy],
                           fillColor=fill, strokeColor=stroke, strokeWidth=1))
        self._lines(text, size, "UI", cx, cy, INK)
        return (cx, cy, w, h)

    def label(self, x, y, text, size=7, color=MUTED, anchor="middle", font="UII"):
        self.d.add(String(x, y, text, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))

    def arrow(self, pts, color=MUTED, width=1.1, head=5):
        flat = [c for p in pts for c in p]
        self.d.add(PolyLine(flat, strokeColor=color, strokeWidth=width))
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        hx1, hy1 = x2 - head * math.cos(a - 0.4), y2 - head * math.sin(a - 0.4)
        hx2, hy2 = x2 - head * math.cos(a + 0.4), y2 - head * math.sin(a + 0.4)
        self.d.add(Polygon([x2, y2, hx1, hy1, hx2, hy2], fillColor=color, strokeColor=color, strokeWidth=0.5))

    # edge points
    @staticmethod
    def top(b): return (b[0], b[1] + b[3] / 2)
    @staticmethod
    def bot(b): return (b[0], b[1] - b[3] / 2)
    @staticmethod
    def left(b): return (b[0] - b[2] / 2, b[1])
    @staticmethod
    def right(b): return (b[0] + b[2] / 2, b[1])

    def down(self, a, b, **k): self.arrow([self.bot(a), self.top(b)], **k)
    def rightto(self, a, b, **k): self.arrow([self.right(a), self.left(b)], **k)

    def band(self, y, h, text, fill):
        self.d.add(Rect(0, y, self.w, h, fillColor=fill, strokeColor=None))
        self.d.add(String(6, y + h - 11, text, fontName="UISB", fontSize=7.5, fillColor=MUTED))


# ---------------------------------------------------------------- figures
def fig_overview():
    BH = 86                       # band height
    fc = FC(CW, 5 * BH)
    w = CW
    names = ["DATA", "AUGMENT", "TRAIN", "SHRINK FOR PHONE", "RUN"]
    for i, n in enumerate(names):
        fc.band((4 - i) * BH, BH, f"{i + 1}  {n}", GRAY_BG if i % 2 == 0 else colors.white)
    cy = lambda i: (4 - i) * BH + 36  # box centre in band i
    X = [w * 0.17, w * 0.5, w * 0.83]
    d = [fc.box(X[0], cy(0), 150, 40, "**ASVspoof 2019 LA\nstudio TTS / VC fakes", fill=colors.white, stroke=LINE),
         fc.box(X[1], cy(0), 150, 40, "**In-the-Wild\nreal politicians & celebs", fill=colors.white, stroke=LINE),
         fc.box(X[2], cy(0), 150, 40, "**DEEP-VOICE\nRVC voice-changer fakes", fill=colors.white, stroke=LINE)]
    aug = fc.box(X[1], cy(1), 330, 40, "**Augmentation  (dfad/augment.py)\nroom echo · noise · MP3 / Opus · phone line · speakerphone chain",
                 fill=AMBER_BG, stroke=AMBER)
    jy = cy(0) - 20 - 12
    for b in d:
        fc.d.add(Line(b[0], cy(0) - 20, b[0], jy, strokeColor=MUTED, strokeWidth=1.1))
    fc.d.add(Line(X[0], jy, X[2], jy, strokeColor=MUTED, strokeWidth=1.1))
    fc.arrow([(X[1], jy), fc.top(aug)])
    m = [fc.box(X[0], cy(2), 150, 40, "**GMM baseline\nMFCC statistics", fill=colors.white, stroke=LINE),
         fc.box(X[1], cy(2), 150, 40, "**Mel-spectrogram CNN\n0.31 M params", fill=colors.white, stroke=LINE),
         fc.box(X[2], cy(2), 150, 40, "**XLS-R 300M + head\nthe TEACHER (most accurate)", fill=VIOLET_BG, stroke=VIOLET)]
    jy = cy(1) - 20 - 12
    fc.d.add(Line(X[1], cy(1) - 20, X[1], jy, strokeColor=MUTED, strokeWidth=1.1))
    fc.d.add(Line(X[0], jy, X[2], jy, strokeColor=MUTED, strokeWidth=1.1))
    for b in m:
        fc.arrow([(b[0], jy), fc.top(b)])
    kd = fc.box(X[2], cy(3), 140, 40, "**Knowledge distillation\nscripts/distill.py", fill=VIOLET_BG, stroke=VIOLET)
    st = fc.box(X[1], cy(3), 140, 40, "**Student model\n0.19 M params, raw audio", fill=GREEN_BG, stroke=GREEN)
    ex = fc.box(X[0], cy(3), 150, 40, "**export_onnx.py\nstudent.onnx + config.json", fill=GREEN_BG, stroke=GREEN)
    fc.down(m[2], kd, color=VIOLET)
    fc.label(X[2] + 5, cy(3) + 30, "soft labels", size=6.8, color=VIOLET, anchor="start")
    fc.arrow([fc.left(kd), fc.right(st)])
    fc.arrow([fc.left(st), fc.right(ex)])
    a = [fc.box(X[0], cy(4), 150, 40, "**Android app\nsame phone, during calls", fill=BLUE_BG, stroke=BLUE),
         fc.box(X[1], cy(4), 150, 40, "**Web app (PWA)\nany browser, offline", fill=BLUE_BG, stroke=BLUE),
         fc.box(X[2], cy(4), 150, 40, "**Laptop demos\nStreamlit · serve.py", fill=BLUE_BG, stroke=BLUE)]
    jy = cy(3) - 20 - 12
    fc.d.add(Line(X[0], cy(3) - 20, X[0], jy, strokeColor=GREEN, strokeWidth=1.1))
    fc.d.add(Line(X[0], jy, X[1], jy, strokeColor=GREEN, strokeWidth=1.1))
    fc.arrow([(X[0], jy), fc.top(a[0])], color=GREEN)
    fc.arrow([(X[1], jy), fc.top(a[1])], color=GREEN)
    fc.label((X[0] + X[1]) / 2, jy + 4, "0.77 MB phone model", size=6.8, color=GREEN)
    ex2 = w - 3
    fc.arrow([fc.right(m[2]), (ex2, cy(2)), (ex2, jy), (X[2] + 40, jy), (X[2] + 40, cy(4) + 20)], color=VIOLET)
    fc.label(ex2 - 6, cy(3) - 4, "big XLS-R model", size=6.8, color=VIOLET, anchor="end")
    return fc.d


def row(specs, height, box_h, size=7.4, gap=16):
    """Evenly spaced boxes left to right, joined by arrows. specs: (text, fill, stroke, width)."""
    fc = FC(CW, height)
    total = sum(s[3] for s in specs)
    gap = (CW - total) / (len(specs) - 1)
    x, boxes = 0, []
    for t, fill, stroke, wd in specs:
        boxes.append(fc.box(x + wd / 2, height / 2, wd, box_h, t, fill=fill, stroke=stroke, size=size))
        x += wd + gap
    for a, c in zip(boxes, boxes[1:]):
        fc.rightto(a, c)
    return fc.d


def fig_xlsr():
    return row([("**4 s audio\n16 kHz", colors.white, LINE, 58),
                ("**XLS-R 300M\nfrozen, fp16\n(not trained)", VIOLET_BG, VIOLET, 76),
                ("**25 layers ×\n[mean, std]\n= 25 × 2048", colors.white, LINE, 72),
                ("**Feature cache\n.npy on disk\n(computed once)", AMBER_BG, AMBER, 78),
                ("**Head (trained)\nlayer weights\n+ MLP → P(fake)", GREEN_BG, GREEN, 80)], 76, 54)


def fig_student():
    return row([("**Raw wave\n64 000\nsamples", colors.white, LINE, 56),
                ("**Normalise\nzero mean,\nunit variance", colors.white, LINE, 62),
                ("**Front conv\nk=129, stride 4\n32 channels", BLUE_BG, BLUE, 70),
                ("**4 residual\nblocks\n64→160 ch", BLUE_BG, BLUE, 64),
                ("**Pool over\ntime: mean\n+ max (320)", colors.white, LINE, 62),
                ("**2 dense\nlayers → sigmoid\n= P(fake)", GREEN_BG, GREEN, 70)], 76, 54, size=7.2)


def fig_kd():
    fc = FC(CW, 150)
    clip = fc.box(55, 75, 90, 40, "**Training clip\n+ true label", fill=colors.white, stroke=LINE)
    tea = fc.box(215, 118, 150, 40, "**Teacher (XLS-R head)\non cached clean features", fill=VIOLET_BG, stroke=VIOLET)
    aug = fc.box(190, 32, 110, 40, "**Random 4 s crop\n+ augmentation", fill=AMBER_BG, stroke=AMBER)
    stu = fc.box(318, 32, 100, 40, "**Student\n(being trained)", fill=GREEN_BG, stroke=GREEN)
    loss = fc.box(445, 75, 120, 64, "**Loss\n(1−α)·BCE(label)\n+ α·T²·BCE(teacher)\nα = 0.5, T = 2", fill=RED_BG, stroke=RED, size=7.6)
    fc.arrow([fc.right(clip), (118, 75), (118, 118), fc.left(tea)])
    fc.arrow([(118, 75), (118, 32), fc.left(aug)])
    fc.rightto(aug, stu)
    fc.arrow([fc.right(tea), (445, 118), fc.top(loss)])
    fc.arrow([fc.right(stu), (445, 32), fc.bot(loss)])
    fc.label(330, 123, "soft label (how fake it sounds)", size=6.8)
    fc.label(395, 22, "prediction", size=6.8)
    return fc.d


def fig_clip():
    fc = FC(CW, 300)
    cx = CW * 0.32
    bw = 220
    steps = [("**Audio arrives\nfile picker · Share from WhatsApp · 8 s recording", colors.white, LINE),
             ("**Decode → 16 kHz mono\nAndroid MediaCodec / browser decodeAudioData", colors.white, LINE),
             ("**Cut into 4 s windows every 2 s\nmax 30, spread over the whole clip", BLUE_BG, BLUE),
             ("**Student scores each window\nONNX Runtime, on the phone (~10-20 ms each)", GREEN_BG, GREEN),
             ("**Average P(fake) over windows", colors.white, LINE)]
    ys = [278, 226, 174, 122, 74]
    boxes = [fc.box(cx, y, bw, 38 if i < 4 else 30, t, fill=f, stroke=s, size=7.8) for i, ((t, f, s), y) in enumerate(zip(steps, ys))]
    for a, c in zip(boxes, boxes[1:]):
        fc.down(a, c)
    dm = fc.diamond(cx, 22, 150, 40, "average ≥ threshold ?")
    fc.down(boxes[-1], dm)
    fake = fc.box(CW * 0.8, 50, 170, 36, "**Likely AI-generated\nshow risk % + safety advice", fill=RED_BG, stroke=RED, size=7.6)
    real = fc.box(CW * 0.8, 8 + 4, 170, 22, "**Likely a real voice  (not a guarantee)", fill=GREEN_BG, stroke=GREEN, size=7.4)
    fc.arrow([fc.right(dm), (CW * 0.6, 22), (CW * 0.6, 50), fc.left(fake)], color=RED)
    fc.arrow([(CW * 0.6, 22), (CW * 0.6, 12), fc.left(real)], color=GREEN)
    fc.label(CW * 0.6 + 4, 54, "yes", size=7, color=RED, anchor="start")
    fc.label(CW * 0.6 + 4, 2, "no", size=7, color=GREEN, anchor="start")
    # side note
    fc.box(CW * 0.8, 200, 165, 92,
           "**Why windows?\nThe model was trained on 4 s.\nLong clips are scored piece by\npiece and averaged, which is more\nstable than one guess. The bars\nin the apps show each window.",
           fill=GRAY_BG, stroke=LINE, size=7.3)
    return fc.d


def fig_live():
    fc = FC(CW, 560)
    cx = CW * 0.36
    g = MUTED
    start = fc.box(cx, 540, 210, 30, "**A call starts\naudio mode = IN_CALL or IN_COMMUNICATION", fill=NAVY, stroke=NAVY, color=colors.white, size=7.6)
    mode = fc.diamond(cx, 482, 150, 50, "Mode set in\nthe app?")
    off = fc.box(CW * 0.84, 500, 110, 26, "Off → do nothing", fill=colors.white, stroke=LINE, size=7.4)
    req = fc.box(CW * 0.84, 464, 140, 34, "On request → show\n\"Check caller\" bubble; wait for tap", fill=colors.white, stroke=LINE, size=7.2)
    mic = fc.box(cx, 418, 210, 32, "**Open the microphone (accessibility service)\nVOICE_RECOGNITION, else MIC source", fill=BLUE_BG, stroke=BLUE, size=7.4)
    opened = fc.diamond(cx, 365, 140, 46, "Mic opened\nand not silent?")
    blocked = fc.box(CW * 0.84, 365, 150, 50, "**Phone blocks the mic\nbubble \"Mic blocked\" +\nnotification with fallbacks", fill=GRAY_BG, stroke=LINE, size=7.2)
    ring = fc.box(cx, 310, 210, 32, "**Read 100 ms at a time into a 4 s ring buffer\n(RAM only, wiped when the call ends)", fill=BLUE_BG, stroke=BLUE, size=7.4)
    hop = fc.diamond(cx, 258, 140, 44, "2 s since last\ncheck?")
    vad = fc.diamond(cx, 204, 140, 44, "Speech in the\nlast 4 s? (VAD)")
    listen = fc.box(CW * 0.84, 204, 120, 26, "bubble \"Listening…\"", fill=colors.white, stroke=LINE, size=7.4)
    score = fc.box(cx, 152, 210, 32, "**Student scores the 4 s window\nbubble shows colour + AI-voice risk %", fill=GREEN_BG, stroke=GREEN, size=7.4)
    red = fc.diamond(cx, 100, 150, 46, "P(fake) ≥ live\nthreshold (red)?")
    reset = fc.box(CW * 0.84, 100, 120, 26, "streak = 0", fill=colors.white, stroke=LINE, size=7.4)
    streak = fc.diamond(cx, 46, 150, 44, "streak + 1 = 3\nreds in a row?")
    alert = fc.box(CW * 0.84, 30, 150, 50, "**ALERT\nred banner at top · vibrate ·\nnotification (hang-up stays usable)", fill=RED_BG, stroke=RED, size=7.2)
    fc.down(start, mode)
    fc.arrow([fc.right(mode), (CW * 0.6, 482), (CW * 0.6, 500), fc.left(off)], color=g)
    fc.arrow([(CW * 0.6, 482), (CW * 0.6, 464), fc.left(req)], color=g)
    fc.label(cx + 6, 449, "Automatic, or bubble tapped", size=6.8, anchor="start")
    fc.arrow([fc.bot(mode), fc.top(mic)])
    fc.arrow([fc.bot(req), (CW * 0.84, 438), (cx + 40, 438), (cx + 40, 434)], color=g)
    fc.down(mic, opened)
    fc.arrow([fc.right(opened), fc.left(blocked)], color=g)
    fc.label(CW * 0.55, 369, "no", size=7)
    fc.down(opened, ring)
    fc.label(cx + 5, 335, "yes", size=7, anchor="start")
    fc.down(ring, hop)
    fc.arrow([fc.left(hop), (cx - 125, 258), (cx - 125, 310), fc.left(ring)], color=g)
    fc.label(cx - 121, 280, "no: keep reading", size=6.8, anchor="start")
    fc.down(hop, vad)
    fc.arrow([fc.right(vad), fc.left(listen)], color=g)
    fc.label(CW * 0.55, 208, "no", size=7)
    fc.arrow([fc.top(listen), (CW * 0.84, 300), (cx + 105, 300)], color=g)
    fc.down(vad, score)
    fc.down(score, red)
    fc.arrow([fc.right(red), fc.left(reset)], color=g)
    fc.label(CW * 0.55, 104, "no", size=7)
    fc.arrow([fc.top(reset), (CW * 0.84, 125), (CW * 0.97, 125), (CW * 0.97, 300), (CW * 0.84 + 0, 300)], color=g)
    fc.down(red, streak)
    fc.label(cx + 5, 72, "yes", size=7, anchor="start")
    fc.arrow([fc.right(streak), (CW * 0.6, 46), (CW * 0.6, 30), fc.left(alert)], color=RED)
    fc.label(CW * 0.6 + 4, 50, "yes", size=7, color=RED, anchor="start")
    fc.arrow([fc.left(streak), (cx - 125, 46), (cx - 125, 258)], color=g)
    fc.label(cx - 121, 52, "no: keep listening", size=6.8, anchor="start")
    return fc.d


def fig_after_call():
    return row([("**Call ends\nmic released,\nbuffer wiped", NAVY_BOX, NAVY, 84),
                ("**Recordings\nfolder chosen\nin the app?", AMBER_BG, AMBER, 84),
                ("**Wait 4 s, then\nfind the newest\nrecording", colors.white, LINE, 90),
                ("**Notification:\n\"Check this call's\nrecording?\"", BLUE_BG, BLUE, 100),
                ("**Tap → normal\nclip check", GREEN_BG, GREEN, 80)], 66, 50, size=7.2)


def fig_risk():
    """P(fake) -> AI-voice risk mapping with the threshold at 50 %."""
    d = Drawing(CW, 160)
    x0, y0, w, h = 60, 34, 230, 110
    t = 0.89
    d.add(Rect(x0, y0, w, h, fillColor=colors.white, strokeColor=LINE))
    for v in (0.25, 0.5, 0.75):
        d.add(Line(x0, y0 + v * h, x0 + w, y0 + v * h, strokeColor=GRAY_BG))
    d.add(Line(x0, y0 + 0.5 * h, x0 + w, y0 + 0.5 * h, strokeColor=RED, strokeWidth=0.6, strokeDashArray=[3, 2]))
    d.add(Line(x0 + t * w, y0, x0 + t * w, y0 + h, strokeColor=RED, strokeWidth=0.6, strokeDashArray=[3, 2]))
    d.add(PolyLine([x0, y0, x0 + t * w, y0 + 0.5 * h, x0 + w, y0 + h], strokeColor=BLUE, strokeWidth=2))
    d.add(Rect(x0, y0 + 0.35 * h, w, 0.15 * h, fillColor=colors.Color(0.98, 0.8, 0.2, 0.15), strokeColor=None))
    d.add(Rect(x0, y0 + 0.5 * h, w, 0.5 * h, fillColor=colors.Color(0.86, 0.15, 0.15, 0.07), strokeColor=None))
    for v, s in ((0, "0"), (0.5, "50%"), (1, "100%")):
        d.add(String(x0 - 6, y0 + v * h - 3, s, fontName="UI", fontSize=7, fillColor=MUTED, textAnchor="end"))
    for v, s in ((0, "0"), (0.5, "0.5"), (t, "threshold"), (1, "1")):
        d.add(String(x0 + v * w, y0 - 11, s, fontName="UI", fontSize=7, fillColor=RED if s == "threshold" else MUTED, textAnchor="middle"))
    d.add(String(x0 + w / 2, 4, "model output P(fake)", fontName="UII", fontSize=7.5, fillColor=MUTED, textAnchor="middle"))
    d.add(String(14, y0 + h / 2, "risk", fontName="UII", fontSize=7.5, fillColor=MUTED, textAnchor="middle"))
    tx = x0 + w + 30
    for i, (c, s) in enumerate(((RED, "red   risk ≥ 50%  = FAKE verdict"), (colors.HexColor("#eab308"), "yellow  35-50%  = some signs"),
                                 (GREEN, "green  < 35%  = low risk"))):
        d.add(Rect(tx, y0 + h - 14 - i * 18, 10, 10, fillColor=c, strokeColor=None))
        d.add(String(tx + 16, y0 + h - 12 - i * 18, s, fontName="UI", fontSize=8, fillColor=INK))
    d.add(String(tx, y0 + 18, "The threshold is wherever the model's", fontName="UII", fontSize=7.5, fillColor=MUTED))
    d.add(String(tx, y0 + 8, "errors balance; it always shows as 50%.", fontName="UII", fontSize=7.5, fillColor=MUTED))
    return d


# ---------------------------------------------------------------- results parsing
def parse_sim(path):
    """{set: {"clip": [rows], "live": [rows]}} from simulate_app.py markdown, plus the header line."""
    if not path.exists():
        return None, None
    out, cur, part = {}, None, None
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].lstrip("# ").strip().replace("App simulation: ", "").replace("`", "")
    for ln in lines:
        if ln.startswith("## "):
            cur = ln[3:].strip()
            out[cur] = {"clip": [], "live": []}
        elif ln.startswith("**Check a clip"):
            part = "clip"
        elif ln.startswith("**Live call"):
            part = "live"
        elif cur and ln.startswith("|") and not ln.startswith("|---") and not ln.startswith("| how") and not ln.startswith("| caller"):
            out[cur][part].append([c.strip() for c in ln.strip("|").split("|")])
    return out, head


def sim_tables(before, after):
    """Side-by-side before/after tables per test set."""
    flow = []
    for s in before:
        part = [P(s, "h2")]
        b = {r[0]: r for r in before[s]["clip"]}
        a = {r[0]: r for r in after[s]["clip"]} if after and s in after else {}
        rows = [["How the audio reached the phone", "Accuracy", "Real flagged as fake", "Fakes caught", "EER"]]
        for cond, r in b.items():
            def pair(i, better_low=False):
                if cond not in a:
                    return r[i]
                o, n = float(r[i].rstrip("%")), float(a[cond][i].rstrip("%"))
                good = (n < o) if better_low else (n > o)
                col = "#16a34a" if good and abs(n - o) >= 1 else ("#dc2626" if (not good) and abs(n - o) >= 1 else "#64748b")
                return f"{r[i]} → <font color='{col}'><b>{a[cond][i]}</b></font>"
            rows.append([cond, pair(1), pair(2, True), pair(3), pair(4, True)])
        part.append(table(rows, [CW * 0.28, CW * 0.18, CW * 0.2, CW * 0.18, CW * 0.16], bold_first=False))
        lb = {r[0]: r for r in before[s]["live"]}
        la = {r[0]: r for r in after[s]["live"]} if after and s in after else {}
        rows = [["Live call on speakerphone", "Calls", "Alert raised", "Median time to alert", "Red windows"]]
        for who, r in lb.items():
            if who in la:
                n = la[who]
                rows.append([who, r[1], f"{r[2]} → <b>{n[2]}</b>", f"{r[3]} → <b>{n[3]}</b>", f"{r[4]} → <b>{n[4]}</b>"])
            else:
                rows.append(r)
        part.append(Spacer(1, 4))
        part.append(table(rows, [CW * 0.28, CW * 0.12, CW * 0.2, CW * 0.22, CW * 0.18]))
        flow.append(KeepTogether(part))
    return flow


# ---------------------------------------------------------------- page decoration
def on_page(c, doc):
    c.saveState()
    if doc.page > 1:
        c.setStrokeColor(LINE)
        c.setLineWidth(0.5)
        c.line(M, H - 12 * mm, W - M, H - 12 * mm)
        c.setFont("UI", 8)
        c.setFillColor(MUTED)
        c.drawString(M, H - 10 * mm, "Call Guard · Deepfake voice detection")
        c.drawRightString(W - M, H - 10 * mm, "How it works, how and why")
        c.drawCentredString(W / 2, 10 * mm, str(doc.page))
    c.restoreState()


def cover(c, doc):
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, H - 105 * mm, W, 105 * mm, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(0, H - 108 * mm, W, 3 * mm, fill=1, stroke=0)
    # shield
    sx, sy = W - M - 34 * mm, H - 72 * mm
    p = c.beginPath()
    p.moveTo(sx, sy + 40 * mm)
    p.lineTo(sx + 16 * mm, sy + 46 * mm)
    p.lineTo(sx + 32 * mm, sy + 40 * mm)
    p.curveTo(sx + 32 * mm, sy + 14 * mm, sx + 24 * mm, sy + 4 * mm, sx + 16 * mm, sy)
    p.curveTo(sx + 8 * mm, sy + 4 * mm, sx, sy + 14 * mm, sx, sy + 40 * mm)
    c.setFillColor(colors.HexColor("#1e3a8a"))
    c.setStrokeColor(colors.HexColor("#60a5fa"))
    c.setLineWidth(1.5)
    c.drawPath(p, fill=1, stroke=1)
    c.setStrokeColor(colors.white)
    c.setLineWidth(2)
    for i, hgt in enumerate((6, 12, 18, 10, 14, 7, 4)):
        x = sx + 6 * mm + i * 3.3 * mm
        c.line(x, sy + 24 * mm - hgt / 2 * mm / 2, x, sy + 24 * mm + hgt / 2 * mm / 2)
    c.setFillColor(colors.white)
    c.setFont("UIB", 30)
    c.drawString(M, H - 45 * mm, "Call Guard")
    c.setFont("UI", 15)
    c.setFillColor(colors.HexColor("#bfdbfe"))
    c.drawString(M, H - 56 * mm, "Detecting AI-generated voices, on the phone,")
    c.drawString(M, H - 63 * mm, "during a call")
    c.setFont("UI", 10)
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.drawString(M, H - 90 * mm, "Project report  ·  what each part does, how it works, and why")
    c.restoreState()


# ---------------------------------------------------------------- content
def build(after_path=None):
    before, bhead = parse_sim(PROJ / "results" / "app_simulation_before.md")
    after, ahead = parse_sim(Path(after_path)) if after_path else (None, None)
    s = []
    s.append(Spacer(1, 98 * mm))
    s.append(P("What this project is", "h1"))
    s.append(P("Voice-cloning tools can now copy a person's voice from a few seconds of audio. Scammers use this to "
               "phone families pretending to be a relative in trouble, or to pose as a bank or a boss. This project "
               "builds a detector that tells a <b>real human voice</b> from an <b>AI-generated</b> one, and puts a "
               "small version of it <b>on the phone itself</b>, so it can check a caller <b>during a live call</b>, "
               "with no audio ever leaving the device."))
    s.append(table([
        ["Part", "What it is", "Runs on"],
        ["Training pipeline<br/><font name='Mono' size='7.5'>dfad/, scripts/</font>",
         "Loads three public datasets, adds real-world degradations, trains and compares four models, distils the best into a tiny phone model.",
         "Laptop GPU (RTX 2050, 4 GB)"],
        ["Android app<br/><font name='Mono' size='7.5'>android/</font>",
         "Checks the caller on the same phone during a call, checks call recordings and shared voice notes.",
         "Android 10+, fully offline"],
        ["Web app (PWA)<br/><font name='Mono' size='7.5'>web/</font>",
         "Check a clip, or listen to a call on speaker from a second device. Installable, works offline.",
         "Any phone or laptop browser"],
        ["Laptop demos<br/><font name='Mono' size='7.5'>app.py, serve.py</font>",
         "Streamlit demo with per-window chart and a 'where the model looked' heatmap; a local server for the big model.",
         "Laptop"],
    ], [CW * 0.22, CW * 0.53, CW * 0.25]))
    s.append(Spacer(1, 8))
    s.append(P("Labels everywhere: <b>0 = real</b> (bona fide), <b>1 = fake</b> (spoof). Every score means "
               "<i>higher = more likely fake</i>.", "small"))
    s.append(PageBreak())

    # ---------------- 1 overview
    s.append(P("1. The big picture", "h1"))
    s.append(P("The project has two halves. On the laptop, a large and accurate model is trained on public "
               "datasets. Because that model is far too big for a phone (300 million parameters, over 1 GB), its "
               "knowledge is <b>distilled</b> into a 0.19-million-parameter Student that is 0.77 MB and scores 4 "
               "seconds of audio in a few milliseconds on a phone CPU. The Student is exported to the ONNX format, "
               "which both the Android app and the web app can run without any internet connection."))
    s.append(fig_overview())
    s.append(P("Figure 1. End-to-end flow, from datasets to the apps.", "caption"))
    s.append(PageBreak())

    # ---------------- 2 components
    s.append(P("2. Which file does what", "h1"))
    s.append(table([
        ["File", "Job", "Why it is built this way"],
        ["<font name='Mono'>config.yaml</font>", "All settings: dataset paths, window length, augmentation probabilities, training hyper-parameters.",
         "One place to change anything; every script reads it."],
        ["<font name='Mono'>dfad/data.py</font>", "Reads the datasets, loads audio as 16 kHz mono, cuts 4 s windows, makes speaker-disjoint splits.",
         "16 kHz is what speech models expect; 4 s is long enough to hear artefacts, short enough for live checking."],
        ["<font name='Mono'>dfad/augment.py</font>", "Real-world degradations: room echo, noise, MP3, Opus, telephone line, speakerphone chain.",
         "Studio-clean training audio does not look like a WhatsApp note or a call on speaker."],
        ["<font name='Mono'>dfad/models.py</font>", "The four models: GMM, mel CNN, XLS-R feature extractor + head, Student.",
         "A classic baseline, a standard CNN, a state-of-the-art model, and a phone model, compared on the same data."],
        ["<font name='Mono'>dfad/cache.py</font>", "Stores XLS-R features on disk (fp16) so the 300M model runs only once per clip.",
         "Training the head then takes minutes instead of days and fits in 4 GB of GPU memory."],
        ["<font name='Mono'>dfad/trainer.py</font>", "Shared training loop: AdamW, cosine learning rate, mixed precision, keeps the epoch with the best dev EER.",
         "Saves the threshold at the dev-set EER inside the checkpoint, so every app uses the right cut-off."],
        ["<font name='Mono'>dfad/metrics.py</font>", "Equal Error Rate (EER), the standard metric for spoof detection.", "Threshold-free, so models are compared fairly."],
        ["<font name='Mono'>dfad/infer.py</font>", "Detector used by the laptop apps: windows, scores, verdict, risk scale, occlusion heatmap.", "One implementation of the scoring rules."],
        ["<font name='Mono'>dfad/calibrate.py</font>", "Works out the live-call threshold from speakerphone-degraded genuine voices.",
         "A loudspeaker raises every score; without it, real callers get flagged."],
        ["<font name='Mono'>scripts/cache_xlsr.py</font>", "Runs frozen XLS-R over a dataset once and writes the feature cache.", "Resumable; about 16 GB on disk."],
        ["<font name='Mono'>scripts/train.py</font>", "Trains the GMM, CNN or XLS-R head.", ""],
        ["<font name='Mono'>scripts/distill.py</font>", "Trains the phone Student with the XLS-R teacher's soft labels.", "Gives the tiny model some of the big model's judgement."],
        ["<font name='Mono'>scripts/export_onnx.py</font>", "Exports the Student to ONNX, tries int8, checks parity, writes config.json, copies to both apps.",
         "Ships only if the exported model matches PyTorch within 0.05."],
        ["<font name='Mono'>scripts/evaluate.py</font>", "EER of every model on every held-out test set → results/table.md.", "The comparison table for the report."],
        ["<font name='Mono'>scripts/simulate_app.py</font>", "Runs the phone model exactly like the apps on degraded test audio and simulated calls.",
         "Measures what a user would actually experience, not just a lab number."],
        ["<font name='Mono'>GuardService.kt</font><br/><font size='7.5' color='#64748b'>Android app</font>", "Accessibility service: detects calls, listens, scores, draws the bubble, raises the alert.",
         "On Android, only an accessibility service may keep using the mic during a call."],
        ["<font name='Mono'>Detector.kt</font><br/><font size='7.5' color='#64748b'>Android app</font>", "Loads student.onnx, cuts windows, scores, risk scale.", "Same maths as the web and Python code."],
        ["<font name='Mono'>AudioFiles.kt</font><br/><font size='7.5' color='#64748b'>Android app</font>", "Decodes any audio file to 16 kHz (with an anti-alias filter), finds the newest call recording.", ""],
        ["<font name='Mono'>MainActivity.kt</font><br/><font size='7.5' color='#64748b'>Android app</font>", "Setup, mode selection, file / Share checks, results.", "Plain Android views, no extra libraries: small APK."],
        ["<font name='Mono'>web/app.js</font>", "Browser version: decode, window, score with ONNX Runtime Web, live meter.", "Plain JavaScript, no build step."],
        ["<font name='Mono'>web/sw.js</font>", "Service worker: caches the app and model for offline use.", ""],
    ], [CW * 0.25, CW * 0.43, CW * 0.32]))
    s.append(PageBreak())

    # ---------------- 3 data
    s.append(P("3. Data: what the models learn from", "h1"))
    s.append(table([
        ["Dataset", "What is in it", "How it is split", "Why it is used"],
        ["ASVspoof 2019 LA", "Studio speech; fakes from 19 TTS and voice-conversion systems.",
         "Official train / dev / eval. Eval uses attacks never seen in training.", "The standard benchmark; tests unseen attack types."],
        ["In-the-Wild", "Real recordings of politicians and celebrities, and deepfakes of them from the internet.",
         "By speaker: 70% train, 10% dev, 20% test. No test speaker is ever heard in training.",
         "Real-world audio quality; tests new people."],
        ["DEEP-VOICE", "Real speeches and RVC voice-changer conversions (e.g. one politician's voice turned into another's).",
         "By person: anything involving the test people (Musk, Trump) is test-only.",
         "Voice changers are the likeliest live-call attack."],
        ["Hindi (planned)", "Volunteer recordings and XTTS-v2 clones of them.", "Test only.", "Checks a language the model never trained on."],
    ], [CW * 0.16, CW * 0.3, CW * 0.29, CW * 0.25]))
    s.append(Spacer(1, 6))
    s.append(callout("If the same person appears in training and testing, a model can learn to recognise <i>voices</i> "
                     "instead of <i>fakeness</i>, and its score looks better than it really is. Every test set here is "
                     "split so the people (or attacks) in it are new to the model.", title="Why speaker-disjoint splits matter"))
    s.append(P("Training mix", "h2"))
    s.append(P("The XLS-R head and the Student train on ASVspoof train + In-the-Wild train + DEEP-VOICE train "
               "(shown 5 times per epoch because it is small but important): 56,079 clips. Dev, used to pick the "
               "best epoch and the threshold, combines the three dev splits: 6,712 clips."))
    s.append(P("4. Augmentation: making training audio look like phone audio", "h1"))
    s.append(P("Every training clip is randomly degraded on the fly (80% of the time), so the model learns to ignore "
               "the channel and focus on signs of synthesis."))
    s.append(table([
        ["Effect", "Simulates", "Chance"],
        ["Speakerphone chain", "The whole in-call path: phone line or VoIP (Opus) → the phone's small loudspeaker "
                               "(bass cut, mild distortion) → room echo → the phone's mic → background noise (10-25 dB SNR).", "35%"],
        ["Room + cheap mic", "Echo of a small room (0.15-0.6 s) and the narrow frequency response of a phone mic.", "40%*"],
        ["Noise", "White or rumbling background noise at 5-30 dB signal-to-noise ratio.", "50%*"],
        ["Telephone line", "8 kHz sampling, 300-3400 Hz band, G.711 mu-law: a classic phone call.", "30%*"],
        ["MP3 / Opus", "Compression of downloads, WhatsApp voice notes and VoIP calls (at most one codec).", "25% / 30%*"],
        ["Gain", "Quieter or louder recording (±6 dB).", "50%"],
    ], [CW * 0.2, CW * 0.67, CW * 0.13]))
    s.append(P("* when the speakerphone chain is not used for that clip.", "small"))
    s.append(callout("The first app simulation showed the phone model was weakest on speakerphone, the main live-call "
                     "case. The individual effects only rarely happened together, so the full chain was almost never "
                     "seen in training. Adding the chain as a single effect fixed that gap.",
                     fg=AMBER, bg=AMBER_BG, title="Why the speakerphone chain was added"))
    s.append(PageBreak())

    # ---------------- 5 models
    s.append(P("5. The models", "h1"))
    s.append(table([
        ["Model", "Input", "Size", "Role", "Dev EER"],
        ["GMM", "MFCC + deltas", "2 × 512 Gaussians", "Classic baseline from ASVspoof challenges", "not trained yet"],
        ["Mel CNN", "Log-mel spectrogram", "0.31 M", "Standard deep-learning baseline", "31.7% (1-epoch smoke run)"],
        ["XLS-R + head", "Raw audio → frozen XLS-R 300M", "300 M frozen + 0.53 M trained", "The teacher; most accurate", "<b>0.66%</b>"],
        ["Student (before)", "Raw audio", "0.19 M (0.77 MB)", "Phone model, trained without teacher", "6.66%"],
        ["Student (new)", "Raw audio", "0.19 M (0.77 MB)", "Phone model, distilled + speakerphone augmentation", "<b>5.57%</b>"],
    ], [CW * 0.16, CW * 0.22, CW * 0.2, CW * 0.26, CW * 0.16]))
    s.append(P("The dev sets differ slightly between runs (the new one includes voice-changer fakes), so the app "
               "simulation in section 9 is the fairer comparison.", "small"))
    s.append(P("XLS-R: the accurate teacher", "h2"))
    s.append(P("XLS-R is a speech model pre-trained by Meta on 436,000 hours of speech in 128 languages. It already "
               "knows what natural speech sounds like in fine detail, which makes synthetic artefacts stand out. "
               "Fine-tuning it needs far more than 4 GB of GPU memory, so it is kept <b>frozen</b>: its 25 internal "
               "layers are summarised (mean and spread over time) once per clip and cached. Only a small head is "
               "trained. It learns which layers matter and classifies the summary."))
    s.append(fig_xlsr())
    s.append(P("Figure 2. XLS-R is run once per clip; the small head trains in about a minute per epoch.", "caption"))
    s.append(P("Student: the phone model", "h2"))
    s.append(P("The Student reads the raw waveform directly. It avoids spectrogram (STFT) operations because they are "
               "not supported by every mobile ONNX runtime. Depthwise-separable convolutions keep it at 0.19 M "
               "parameters, small enough to run every 2 seconds during a call without draining the battery."))
    s.append(fig_student())
    s.append(P("Figure 3. Student architecture. Each block also has a skip connection and divides the time axis by 3.", "caption"))
    s.append(KeepTogether([P("Knowledge distillation", "h2"), P("A tiny model trained only on right/wrong labels learns little. With distillation it also copies the "
               "teacher's <i>degree</i> of suspicion for each clip: a fake the teacher finds very obvious and one it "
               "barely catches give different targets. The temperature T softens the teacher's answer so these "
               "differences are visible."), fig_kd(),
        P("Figure 4. Distillation: the teacher's soft label and the true label are combined in the loss.", "caption")]))

    # ---------------- 6 measuring
    s.append(P("6. How accuracy is measured", "h1"))
    s.append(P("<b>Equal Error Rate (EER).</b> Any detector can trade false alarms (real voices called fake) against "
               "misses (fakes called real) by moving its threshold. EER is the error rate at the threshold where both "
               "are equal. Lower is better; 0% is perfect, 50% is guessing. It is the standard metric for spoof "
               "detection because it does not depend on a chosen threshold."))
    s.append(P("<b>The threshold.</b> After every training epoch, the threshold at the dev-set EER is stored in the "
               "checkpoint and later in <font name='Mono'>config.json</font>, so the apps use exactly the cut-off "
               "the model was validated with."))
    s.append(P("<b>The AI-voice risk scale shown to users.</b> A model's threshold can be anywhere (0.89 for the "
               "new Student), so showing the raw probability is confusing: \"89%\" would actually mean <i>real</i>. "
               "Every app therefore stretches the scale so the threshold always sits at 50%:"))
    s.append(fig_risk())
    s.append(P("Figure 5. Raw model output to the risk % shown in all apps.", "caption"))
    s.append(P("<b>Two thresholds.</b> <font name='Mono'>threshold</font> is used for checking files. "
               "<font name='Mono'>live_threshold</font> is used while listening to a call on speaker. It is set on "
               "genuine dev voices played through the speakerphone chain, so that only 15% of genuine windows turn red (chosen with a sweep over simulated calls). "
               "The loudspeaker raises every score, so without this, real callers were often flagged."))

    # ---------------- 7 apps
    s.append(KeepTogether([P("7. How the apps check a voice", "h1"),
                           P("Check a clip (Android, web, Streamlit)", "h2"), fig_clip(),
                           P("Figure 6. Checking a file, voice note or recording.", "caption")]))
    s.append(P("Choosing a mode", "h2"))
    s.append(table([
        ["Mode", "What happens during a call", "Good for"],
        ["Off (default)", "Nothing. The mic is never opened.", "Privacy by default."],
        ["On request", "A \"Check caller\" bubble appears; listening starts only when tapped. Tap again to stop.", "Checking only suspicious calls."],
        ["Automatic", "Listening starts as soon as a call is answered.", "Vulnerable users (e.g. elderly relatives)."],
    ], [CW * 0.18, CW * 0.55, CW * 0.27]))
    s.append(KeepTogether([P("Live check during a call (Android)", "h2"), P("Browsers and ordinary apps receive silence from the microphone while a call is active. Android does "
               "let an <b>accessibility service</b> keep recording during a call (the method call-recorder apps use). "
               "Call Guard's service reads no screen content; it only watches the audio mode to know when a call is "
               "active, draws a small draggable bubble, and listens. The call must be on <b>speaker</b> so the "
               "phone's own mic hears the caller."), fig_live(),
        P("Figure 7. Live checking. Requiring 3 red windows in a row (about 6 s of suspicious speech) "
          "avoids alarms from a single odd window.", "caption")]))
    s.append(KeepTogether([P("After the call", "h2"), fig_after_call(),
        P("Figure 8. Fallback when the phone blocks live listening: check the dialer's own call recording.", "caption")]))
    s.append(P("The web app", "h2"))
    s.append(P("The same Student runs in the browser through ONNX Runtime Web (WebAssembly). <b>Check a clip</b> works "
               "on the same phone. <b>Call Guard live</b> uses the identical ring-buffer, VAD, live-threshold and "
               "3-in-a-row logic, but must run on a <b>second device</b> next to the speakerphone, because phones mute "
               "browsers during calls. On iPhone this is the only live option: Apple gives no app access to call audio. "
               "When served by <font name='Mono'>serve.py</font> on the laptop, the page can switch to the big XLS-R model."))
    s.append(P("Privacy by design", "h2"))
    s.extend(bullets([
        "The Android app has <b>no internet permission</b>: audio physically cannot leave the phone.",
        "Live audio exists only in a 4-second RAM buffer that is overwritten every 100 ms and wiped when the call ends.",
        "Checked files are zeroed in memory after analysis; nothing is saved.",
        "The accessibility service is configured to receive no screen content.",
        "The web app runs the model locally; the only network use is the first download of the page and model.",
    ]))

    # ---------------- 8 engineering choices
    s.append(P("8. Key engineering choices", "h1"))
    s.append(table([
        ["Choice", "Reason"],
        ["4 s windows, every 2 s", "Long enough for the model to hear artefacts; frequent enough that a live warning comes within about 10 s."],
        ["Up to 30 windows, spread over the clip", "Keeps long recordings fast while still covering the whole call, not just its start."],
        ["Energy VAD (> -45 dBFS in 25% of 20 ms frames)", "Silence and pauses carry no evidence; scoring them would create random alerts."],
        ["3 red windows in a row", "One window can be odd; three in a row is consistent evidence."],
        ["Digital silence = blocked", "If the mic returns exact zeros, the phone is muting other apps; the user is told and offered fallbacks."],
        ["Alert as a top banner", "The advice is \"hang up and call back\", so the dialer's hang-up button must stay usable."],
        ["Accessibility service, no AndroidX", "The only same-phone way to hear a call; framework-only code keeps the app small."],
        ["Anti-alias filter before resampling", "Phone recordings are often 44.1/48 kHz; without filtering, high frequencies fold into the band the model inspects."],
        ["int8 only if parity < 0.05", "Quantisation can change predictions; the fp32 model (0.77 MB) is shipped when int8 is not faithful enough."],
    ], [CW * 0.33, CW * 0.67]))
    s.append(PageBreak())

    # ---------------- 9 results
    s.append(P("9. Results: the phone model in app conditions", "h1"))
    s.append(P("<font name='Mono'>scripts/simulate_app.py</font> runs the shipped model exactly as the apps do, on "
               "held-out test clips that reach the phone in five ways, and on simulated 45-second calls heard over "
               "speakerphone. <b>Accuracy</b> is balanced (average of real-correct and fake-correct). "
               "<b>Real flagged</b> is the false-alarm rate on genuine voices; lower is better. "
               "<b>EER</b> shows how well real and fake are separated regardless of threshold; lower is better."))
    if after:
        s.append(callout(f"Each cell shows <b>before → after</b>. Before: the model previously in the apps "
                         f"(<i>{bhead}</i>). After: the new distilled, speakerphone-trained model (<i>{ahead}</i>). "
                         "Green = better by at least 1 point, red = worse.", title="Reading the tables"))
    else:
        s.append(callout("These are the results of the model that was in the apps before retraining. The retrained "
                         "model's results are added when its simulation finishes.", fg=AMBER, bg=AMBER_BG,
                         title="Before retraining"))
    s.append(Spacer(1, 4))
    s.extend(sim_tables(before, after))
    s.append(KeepTogether([
        P("Choosing the live-call threshold", "h2"),
        P("The live threshold trades false alarms on genuine callers against caught AI callers. The same "
          "simulated speakerphone calls were scored once with the new model and checked at several settings "
          "(target = share of genuine dev windows allowed to turn red). 15% was chosen."),
        table([
            ["Setting", "Genuine callers alerted<br/>ITW / DEEP-VOICE / ASVspoof", "AI callers alerted<br/>ITW / DEEP-VOICE / ASVspoof"],
            ["Old app model", "13% / 30% / 30%", "67% / 80% / 83%"],
            ["5% target", "3% / 0% / 3%", "57% / 27% / 53%"],
            ["10% target", "7% / 0% / 10%", "70% / 50% / 70%"],
            ["<b>15% target (shipped)</b>", "<b>10% / 7% / 10%</b>", "<b>73% / 73% / 80%</b>"],
            ["20% target", "17% / 17% / 33%", "77% / 87% / 87%"],
        ], [CW * 0.26, CW * 0.37, CW * 0.37]),
        P("30 calls per group, so each figure is roughly ±8 points.", "small"),
    ]))
    s.append(Spacer(1, 6))
    s.append(P("Note: training uses similar degradations to these simulated conditions, so the speakerphone numbers "
               "are realistic but somewhat optimistic. A test with real phones, calls and loudspeakers is the final check.", "small"))
    s.append(PageBreak())

    # ---------------- 10 limits
    s.append(P("10. Limitations", "h1"))
    s.extend(bullets([
        "<b>iPhone:</b> no app can hear call audio. Use Check a clip after the call (iOS 18+ can record calls) or a second device.",
        "<b>Android in-call listening depends on the phone.</b> Some manufacturers mute other apps during calls; the app detects this and offers the fallbacks. It also needs speakerphone.",
        "<b>New cloning tools</b> may fool the model until it is retrained on them. Voice-changer (RVC) fakes are the hardest case.",
        "<b>Heavy noise and poor lines</b> reduce accuracy, as the results show.",
        "<b>The Student is less accurate than XLS-R</b> (it is about 1,500 times smaller). The laptop server can use XLS-R when accuracy matters more than portability.",
        "<b>A warning is not proof, and no warning is not a guarantee.</b> The app's advice is always to verify through another channel.",
        "<b>The Hindi test set</b> is still to be recorded; its results will be indicative because it is small.",
    ]))
    s.append(P("11. How to run it", "h1"))
    s.append(P("All commands run from the project folder in Command Prompt or PowerShell.", "small"))
    s.append(code(
        "cd C:\\Users\\HP\\OneDrive\\Desktop\\audio\n"
        ".venv\\Scripts\\python.exe -m pytest tests -q                    # self-tests\n\n"
        "# training (laptop GPU)\n"
        ".venv\\Scripts\\python.exe scripts\\cache_xlsr.py asvspoof_train itw_train deepvoice_train asvspoof_dev itw_dev deepvoice_dev\n"
        ".venv\\Scripts\\python.exe scripts\\train.py --model w2v           # the XLS-R teacher\n"
        ".venv\\Scripts\\python.exe scripts\\distill.py                     # the phone Student\n"
        ".venv\\Scripts\\python.exe scripts\\export_onnx.py --int8          # -> web/ and android/\n\n"
        "# measuring\n"
        ".venv\\Scripts\\python.exe scripts\\evaluate.py                    # EER table\n"
        ".venv\\Scripts\\python.exe scripts\\simulate_app.py                # app-condition simulation\n\n"
        "# apps\n"
        ".venv\\Scripts\\python.exe serve.py                               # website at http://localhost:8000\n"
        ".venv\\Scripts\\streamlit.exe run app.py                          # Streamlit demo\n"
        "cd android && C:\\android\\gradle\\gradle-8.10.2\\bin\\gradle.bat assembleRelease   # Android APK"))
    s.append(P("Tools used", "h2"))
    s.append(table([
        ["Area", "Tools"],
        ["Training", "Python 3.11, PyTorch 2 (CUDA), Hugging Face Transformers (XLS-R), scikit-learn (GMM, ROC/EER), librosa, SciPy, soundfile (MP3/Opus without ffmpeg)"],
        ["Phone runtime", "ONNX + ONNX Runtime (Android 1.20, Web 1.20 WebAssembly), int8 quantisation tools"],
        ["Android", "Kotlin, Android framework only (AccessibilityService, AudioRecord, MediaCodec), Gradle 8.10, minSdk 29"],
        ["Web", "Plain HTML/CSS/JS, AudioWorklet, Service Worker (offline), Web Audio API"],
        ["Demos", "Streamlit, Matplotlib, Python http.server"],
    ], [CW * 0.2, CW * 0.8]))

    OUT.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=17 * mm, bottomMargin=16 * mm,
                            title="Call Guard: deepfake voice detection", author="Call Guard project",
                            subject="How the system works, how and why")
    doc.build(s, onFirstPage=lambda c, d: (cover(c, d), on_page(c, d)), onLaterPages=on_page)
    print(OUT)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
