# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Draw PSPBird's XMB art: ``icon0.png`` (144x80) and ``pic1.png`` (480x272),
the 24-bit PNGs ``Psp.toml`` hands to pack-pbp. Pure vector drawing, no
photo, so it is licence-free and regenerable::

    uv run examples/birdnet/make_icon.py

Rendered at 4x and downsampled for antialiasing. A crested songbird in
silhouette (the fixture is a cardinal) singing a few sound-wave arcs on a
dusk gradient, plus the wordmark on the background.
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "device")
SS = 4  # supersampling factor

BIRD = (255, 92, 62)       # coral-red, reads as a cardinal
BIRD_DARK = (176, 40, 30)
CREAM = (255, 240, 220)
WAVE = (120, 230, 255)
SKY_TOP = (18, 22, 64)
SKY_BOTTOM = (10, 90, 110)
HORIZON = (255, 150, 70)


def gradient(w: int, h: int) -> Image.Image:
  img = Image.new("RGB", (w, h))
  px = img.load()
  for y in range(h):
    t = y / max(h - 1, 1)
    # Dusk: indigo at the top, teal below, a warm band near the bottom.
    base = tuple(int(SKY_TOP[i] * (1 - t) + SKY_BOTTOM[i] * t) for i in range(3))
    glow = max(0.0, 1 - abs(t - 0.82) / 0.22) ** 2 * 0.55
    c = tuple(min(255, int(base[i] * (1 - glow) + HORIZON[i] * glow)) for i in range(3))
    for x in range(w):
      px[x, y] = c
  return img


def ellipse_pts(cx: float, cy: float, rx: float, ry: float, angle_deg: float, n: int = 48):
  """Polygon points of a rotated ellipse (PIL's ellipse() is axis-aligned)."""
  a = math.radians(angle_deg)
  pts = []
  for i in range(n):
    t = 2 * math.pi * i / n
    x, y = rx * math.cos(t), ry * math.sin(t)
    pts.append((cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)))
  return pts


def bird(draw: ImageDraw.ImageDraw, ox: float, oy: float, s: float) -> None:
  """A perched crested songbird facing right. ``s`` is the unit scale;
  the bird spans roughly 1.0 x 0.95 units from (ox, oy) at its bottom-left."""
  P = lambda x, y: (ox + x * s, oy - y * s)  # y up
  E = lambda cx, cy, rx, ry, ang: [P(x, y) for x, y in ellipse_pts(cx, cy, rx, ry, ang)]
  dark = (60, 40, 40)
  # Tail: a slim tapered sweep down-left.
  draw.polygon([P(0.02, 0.10), P(0.08, 0.04), P(0.38, 0.28), P(0.34, 0.36)], fill=BIRD_DARK)
  # Body, tilted up toward the head.
  draw.polygon(E(0.48, 0.40, 0.30, 0.22, 18), fill=BIRD)
  # Head.
  draw.ellipse([P(0.58, 0.82), P(0.88, 0.52)], fill=BIRD)
  # Crest: two pointed spikes leaning back.
  draw.polygon([P(0.60, 0.76), P(0.64, 0.94), P(0.70, 0.80)], fill=BIRD)
  draw.polygon([P(0.66, 0.78), P(0.74, 1.00), P(0.80, 0.78)], fill=BIRD)
  # Beak.
  draw.polygon([P(0.86, 0.70), P(1.00, 0.67), P(0.86, 0.61)], fill=CREAM)
  # Black mask, then the eye.
  draw.polygon(E(0.80, 0.66, 0.09, 0.07, 0), fill=(40, 20, 25))
  draw.ellipse([P(0.78, 0.71), P(0.84, 0.64)], fill=CREAM)
  draw.ellipse([P(0.80, 0.69), P(0.83, 0.66)], fill=(20, 10, 15))
  # Wing: a folded ellipse along the back.
  draw.polygon(E(0.46, 0.40, 0.19, 0.09, 22), fill=BIRD_DARK)
  # Perch and feet.
  draw.line([P(-0.10, 0.12), P(0.84, 0.08)], fill=dark, width=int(0.05 * s))
  for x in (0.42, 0.54):
    draw.line([P(x, 0.20), P(x - 0.02, 0.09)], fill=dark, width=int(0.025 * s))


def waves(draw: ImageDraw.ImageDraw, cx: float, cy: float, s: float) -> None:
  """Three sound-wave arcs radiating from the beak, fading outward."""
  for i, alpha in enumerate((230, 170, 110)):
    r = s * (0.16 + 0.14 * i)
    box = [cx - r, cy - r, cx + r, cy + r]
    draw.arc(box, start=-38, end=38, fill=WAVE + (alpha,), width=max(2, int(0.035 * s)))


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  for path in (
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
      "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
      "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
  ):
    if os.path.exists(path):
      return ImageFont.truetype(path, size)
  return ImageFont.load_default()


def render(w: int, h: int, with_wordmark: bool) -> Image.Image:
  W, H = w * SS, h * SS
  img = gradient(W, H).convert("RGBA")
  layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
  d = ImageDraw.Draw(layer)
  s = H * (0.66 if with_wordmark else 0.80)
  ox = W * (0.05 if with_wordmark else 0.16)
  oy = H * 0.94
  bird(d, ox, oy, s)
  waves(d, ox + 1.00 * s, oy - 0.65 * s, s)
  if with_wordmark:
    f = font(int(H * 0.20))
    text = "PSPBird"
    tw = d.textlength(text, font=f)
    x = W * 0.95 - tw
    y = H * 0.24
    d.text((x + SS, y + SS), text, font=f, fill=(0, 0, 0, 140))
    d.text((x, y), text, font=f, fill=CREAM + (255,))
    f2 = font(int(H * 0.062))
    sub = "BirdNET on a PSP"
    d.text((W * 0.95 - d.textlength(sub, font=f2), y + H * 0.24), sub, font=f2, fill=WAVE + (220,))
  img.alpha_composite(layer)
  return img.convert("RGB").resize((w, h), Image.LANCZOS)


def main() -> None:
  os.makedirs(OUT, exist_ok=True)
  icon = render(144, 80, with_wordmark=False)
  icon.save(os.path.join(OUT, "icon0.png"), optimize=True)
  pic = render(480, 272, with_wordmark=True)
  pic.save(os.path.join(OUT, "pic1.png"), optimize=True)
  print(f"wrote {OUT}/icon0.png (144x80) and pic1.png (480x272)")


if __name__ == "__main__":
  main()
