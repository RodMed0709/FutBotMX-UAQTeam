"""Overlay de espectador (ronda de entregable de eventos) — el showpiece visual.

Compone un **video de espectador** (mp4) a partir del ``tracks_json`` de cámara superior:
el video del partido dentro de un lienzo con **márgenes** + paneles:

- **marcador** por portería (color de cada portería), que sube con cada gol;
- **banner** "¡Goool! Portería {color}" que se desliza de izquierda a derecha;
- **minimapa** (cenital con estela) y **heatmap** (acumulado) en lados opuestos;
- **panel de métricas** (posesión / control);
- **lista dinámica** de eventos (tiros, fueras, lack-of-progress, pushing) con tope.

Configurable: ``layout ∈ {1,2}`` (default 2 = paneles laterales) y ``goal_source ∈
{"strict","geometric"}`` (default strict = ``event_shot_goal``). **Consume** los módulos de
eventos/minimapa/heatmap; **no** toca ``demo_overlay``/``track_overlay`` (quedan para
mosaico/depuración). Render **incremental** frame a frame; CPU local, sin GPU. Si la homografía
no es fiable corre en **modo degradado** (sin minimapa/heatmap). ``cv2`` perezoso.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.demo_overlay import _accumulate, _live_heatmap_state
from src.core.event_field_violations import compute_field_violations
from src.core.event_goal_geometric import compute_geometric_goals
from src.core.event_possession_refine import compute_possession_refine
from src.core.event_shot_goal import compute_shot_vs_goal
from src.core.events_core import BALL_CLASSES, ROBOT_CLASS, load_frame_objects
from src.core.events_schema import events_paths
from src.core.frame_extraction import get_video_fps, iter_frames
from src.core.metric_heatmap import render_heatmap
from src.core.metric_positions import compute_metric_positions
from src.core.minimap import MinimapRenderer

DEFAULT_BANNER_SECS = 2.5
DEFAULT_MAX_ITEMS = 6
DEFAULT_MARGIN_PX = 220
DEFAULT_TRAJ_WINDOW = 60
DEFAULT_BIN_CM = 5.0
DEFAULT_SIGMA_CM = 8.0

_ZONE_COLOR_BGR = {"yellow": (0, 215, 255), "blue": (220, 90, 30)}  # color de cada portería
_DARK = (24, 24, 24)
_WHITE = (245, 245, 245)


@dataclass
class BroadcastResult:
    video: Path
    sample_png: Path | None
    resumen: dict


# --- componentes de dibujo (cv2, BGR) -----------------------------------------

def _fit_box(img: np.ndarray, bw: int, bh: int) -> np.ndarray:
    """Escala ``img`` para caber en ``(bw, bh)`` preservando aspecto."""
    import cv2

    h, w = img.shape[:2]
    s = min(bw / w, bh / h)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))


def _paste(panel: np.ndarray, img: np.ndarray, y0: int) -> None:
    """Pega ``img`` centrada horizontalmente en ``panel`` a partir de ``y0`` (in place)."""
    ph, pw = panel.shape[:2]
    ih, iw = img.shape[:2]
    x = max(0, (pw - iw) // 2)
    yh = min(ih, ph - y0)
    xw = min(iw, pw - x)
    if yh > 0 and xw > 0:
        panel[y0:y0 + yh, x:x + xw] = img[:yh, :xw]


def _overlay(canvas: np.ndarray, img: np.ndarray, x: int, y: int, alpha: float) -> None:
    """Mezcla ``img`` sobre ``canvas`` en ``(x, y)`` con transparencia ``alpha`` (in place)."""
    h, w = img.shape[:2]
    H, W = canvas.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        img = img[:max(0, min(h, H - y)), :max(0, min(w, W - x))]
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return
    roi = canvas[y:y + h, x:x + w].astype(float)
    canvas[y:y + h, x:x + w] = (roi * (1 - alpha) + img.astype(float) * alpha).astype(np.uint8)


def _draw_scoreboard(width: int, yellow_n: int, blue_n: int, h: int = 84) -> np.ndarray:
    import cv2

    bar = np.full((h, width, 3), _DARK, dtype=np.uint8)
    cx = width // 2
    f, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3
    cv2.putText(bar, "AMARILLA", (cx - 320, h // 2 + 12), f, 0.8, _ZONE_COLOR_BGR["yellow"], 2,
                cv2.LINE_AA)
    cv2.putText(bar, f"{yellow_n}", (cx - 110, h // 2 + 16), f, sc, _ZONE_COLOR_BGR["yellow"],
                th, cv2.LINE_AA)
    cv2.putText(bar, "-", (cx - 14, h // 2 + 14), f, sc, _WHITE, th, cv2.LINE_AA)
    cv2.putText(bar, f"{blue_n}", (cx + 70, h // 2 + 16), f, sc, _ZONE_COLOR_BGR["blue"], th,
                cv2.LINE_AA)
    cv2.putText(bar, "AZUL", (cx + 170, h // 2 + 12), f, 0.8, _ZONE_COLOR_BGR["blue"], 2,
                cv2.LINE_AA)
    return bar


def _draw_metrics_panel(w: int, h: int, mdata: dict) -> np.ndarray:
    import cv2

    p = np.full((h, w, 3), (32, 32, 32), dtype=np.uint8)
    cv2.putText(p, "POSESION / CONTROL", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1,
                cv2.LINE_AA)

    def bar(y: int, label: str, pct: float, color) -> None:
        cv2.putText(p, f"{label}: {pct:.0f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1,
                    cv2.LINE_AA)
        cv2.rectangle(p, (10, y + 6), (w - 12, y + 20), (70, 70, 70), -1)
        cv2.rectangle(p, (10, y + 6), (10 + int((w - 22) * pct / 100.0), y + 20), color, -1)

    bar(60, "Posesion", mdata["pct_pos"], (90, 200, 90))
    bar(105, "Control", mdata["pct_ctrl"], (90, 160, 240))
    now_pos = f"#{mdata['now_pos']}" if mdata["now_pos"] is not None else "-"
    now_ctrl = f"#{mdata['now_ctrl']}" if mdata["now_ctrl"] is not None else "-"
    cv2.putText(p, f"Ahora: pos {now_pos}  ctrl {now_ctrl}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return p


def _draw_event_feed(w: int, h: int, items: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    import cv2

    p = np.full((h, w, 3), (32, 32, 32), dtype=np.uint8)
    cv2.putText(p, "EVENTOS", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1, cv2.LINE_AA)
    y = 52
    for text, color in items:
        cv2.rectangle(p, (10, y - 11), (22, y + 1), color, -1)
        cv2.putText(p, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)
        y += 28
        if y > h - 8:
            break
    return p


def _draw_goal_banner(vid: np.ndarray, texto: str, progress: float) -> None:
    """Dibuja el banner deslizante (in place) sobre ``vid`` (BGR). ``progress`` ∈ [0,1]."""
    import cv2

    h, w = vid.shape[:2]
    bh = 70
    y0 = h // 2 - bh // 2
    ribbon = vid[y0:y0 + bh, :].copy()
    cv2.rectangle(ribbon, (0, 0), (w, bh), (40, 40, 220), -1)
    vid[y0:y0 + bh, :] = (vid[y0:y0 + bh, :].astype(float) * 0.25
                          + ribbon.astype(float) * 0.75).astype(np.uint8)
    (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
    x = int(-tw + progress * (w + tw))  # entra por la izquierda, sale por la derecha
    cv2.putText(vid, texto, (x, y0 + 48), cv2.FONT_HERSHEY_DUPLEX, 1.2, _WHITE, 3, cv2.LINE_AA)


# --- layouts -------------------------------------------------------------------

def _compose_layout2(vid, score, mdata, items, mini, heat, banner, margin) -> np.ndarray:
    th = 720
    vid = _fit_box(vid, 100000, th)
    if banner is not None:
        _draw_goal_banner(vid, *banner)
    vh, vw = vid.shape[:2]
    width = margin + vw + margin
    sb = _draw_scoreboard(width, score[0], score[1])
    canvas = np.full((sb.shape[0] + vh, width, 3), (18, 18, 18), dtype=np.uint8)
    canvas[0:sb.shape[0], :] = sb
    top = sb.shape[0]
    # margen izquierdo: métricas (arriba) + lista de eventos (abajo)
    left = np.full((vh, margin, 3), (28, 28, 28), dtype=np.uint8)
    left[0:vh // 2] = _draw_metrics_panel(margin, vh // 2, mdata)
    left[vh // 2:] = _draw_event_feed(margin, vh - vh // 2, items)
    canvas[top:top + vh, 0:margin] = left
    # centro: video
    canvas[top:top + vh, margin:margin + vw] = vid
    # margen derecho: minimapa (arriba) + heatmap (abajo), lados opuestos a las métricas
    right = np.full((vh, margin, 3), (28, 28, 28), dtype=np.uint8)
    if mini is not None:
        _paste(right, _fit_box(mini, margin - 10, vh // 2 - 10), 5)
    if heat is not None:
        _paste(right, _fit_box(heat, margin - 10, vh // 2 - 10), vh // 2 + 5)
    canvas[top:top + vh, margin + vw:] = right
    return canvas


def _compose_layout1(vid, score, mdata, items, mini, heat, banner, margin) -> np.ndarray:
    th = 900
    vid = _fit_box(vid, 100000, th)
    if banner is not None:
        _draw_goal_banner(vid, *banner)
    canvas = vid.copy()
    H, W = canvas.shape[:2]
    pw = int(W * 0.24)
    sb = _draw_scoreboard(int(W * 0.5), score[0], score[1], h=70)
    _overlay(canvas, sb, (W - sb.shape[1]) // 2, 8, 0.8)
    if mini is not None:
        _overlay(canvas, _fit_box(mini, pw, int(H * 0.32)), 8, 86, 0.85)
    if heat is not None:
        hh = _fit_box(heat, pw, int(H * 0.32))
        _overlay(canvas, hh, W - hh.shape[1] - 8, 86, 0.85)
    mp = _draw_metrics_panel(pw, int(H * 0.22), mdata)
    _overlay(canvas, mp, 8, H - mp.shape[0] - 8, 0.7)
    fp = _draw_event_feed(pw, int(H * 0.30), items)
    _overlay(canvas, fp, W - fp.shape[1] - 8, H - fp.shape[0] - 8, 0.7)
    return canvas


# --- precómputo ----------------------------------------------------------------

def _event_items(shot, viol) -> list[tuple[int, str, tuple[int, int, int]]]:
    """Lista de eventos para el feed: ``(frame_inicio, texto, color_bgr)`` ordenada."""
    items: list[tuple[int, str, tuple[int, int, int]]] = []
    for e in shot.eventos:
        if e.tipo == "tiro":
            items.append((e.frame_inicio, f"Tiro a {e.zona}", (0, 165, 255)))
        else:
            items.append((e.frame_inicio, f"GOL {e.zona}", (60, 220, 60)))
    for e in viol.eventos:
        if e.tipo == "fuera":
            txt = (f"Area chica #{e.obj_ids[0]}" if e.causa == "area_chica"
                   else f"Fuera #{e.obj_ids[0]}")
            items.append((e.frame_inicio, txt, (0, 90, 230)))
        elif e.tipo == "lack_of_progress":
            items.append((e.frame_inicio, "Sin progreso", (150, 150, 150)))
        elif e.tipo == "pushing":
            items.append((e.frame_inicio, f"Pushing #{e.obj_ids[0]}-#{e.obj_ids[1]}",
                          (200, 0, 150)))
    items.sort(key=lambda t: t[0])
    return items


def _mini_by_frame(metric) -> dict[int, list[tuple[int, str, float, float]]]:
    out: dict[int, list[tuple[int, str, float, float]]] = {}
    for p in metric.posiciones:
        if p.xy_cm is not None and (p.cls in BALL_CLASSES or p.cls == ROBOT_CLASS):
            out.setdefault(p.frame_index, []).append((p.obj_id, p.cls, p.xy_cm[0], p.xy_cm[1]))
    return out


# --- API pública ---------------------------------------------------------------

def render_broadcast_overlay(
    tracks_json: str | Path,
    *,
    layout: int = 2,
    goal_source: str = "strict",
    banner_secs: float = DEFAULT_BANNER_SECS,
    max_items: int = DEFAULT_MAX_ITEMS,
    margin_px: int = DEFAULT_MARGIN_PX,
    trajectory_window: int = DEFAULT_TRAJ_WINDOW,
    bin_cm: float = DEFAULT_BIN_CM,
    sigma_cm: float = DEFAULT_SIGMA_CM,
    out_fps: float | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    progress: bool = True,
) -> BroadcastResult:
    """Renderiza el video de espectador. ``layout`` 1|2 (default 2); ``goal_source`` strict|geometric."""
    import cv2

    from src.core.video_writer import open_video_writer

    if layout not in (1, 2):
        raise ValueError(f"layout inválido: {layout!r} (usa 1 o 2)")
    if goal_source not in ("strict", "geometric"):
        raise ValueError(f"goal_source inválido: {goal_source!r} (usa 'strict' o 'geometric')")

    tracks_json = Path(tracks_json)
    clip = tracks_json.parent / f"{tracks_json.stem}.mp4"
    if not clip.exists():
        raise FileNotFoundError(f"no está el clip del partido: {clip}")
    fps = out_fps or get_video_fps(clip)
    by_frame = load_frame_objects(tracks_json)

    # --- precómputo de eventos (homografía una vez) ---
    degradado = False
    metric = None
    try:
        metric = compute_metric_positions(tracks_json)
    except Exception:
        degradado = True
    shot = compute_shot_vs_goal(metric if metric is not None else tracks_json, route="cm") \
        if metric is not None else compute_shot_vs_goal(tracks_json, route="px")
    if goal_source == "geometric" and metric is not None:
        geo = compute_geometric_goals(metric, fps=fps)
        goals = [(e.frame_inicio, e.zona) for e in geo.eventos]
    else:
        goals = [(e.frame_inicio, e.zona) for e in shot.eventos if e.tipo == "gol"]
    poss = compute_possession_refine(by_frame, fps=fps)
    viol = compute_field_violations(tracks_json, fps=fps)

    if metric is None or not any(p.xy_cm is not None for p in metric.posiciones):
        degradado = True
    mini_by_frame = _mini_by_frame(metric) if not degradado else {}
    if not degradado:
        grid, hm_by_frame = _live_heatmap_state(metric, bin_cm)
        renderer = MinimapRenderer(trail_len=trajectory_window)

    items_all = _event_items(shot, viol)
    banner_frames = max(1, int(round(banner_secs * (fps or 30.0))))
    compose = _compose_layout2 if layout == 2 else _compose_layout1

    out = events_paths(tracks_json.stem, "broadcast", "mp4")
    sample_png = None
    sample_frame = (goals[0][0] + banner_frames // 2) if goals else None

    bar = None
    if progress:
        from tqdm.auto import tqdm
        total = max_frames or by_frame and (max(by_frame) + 1)
        bar = tqdm(total=total, desc=f"broadcast L{layout}", unit="f", leave=False)

    n = 0  # frames escritos
    final_score = (0, 0)
    with open_video_writer(out, fps=fps) as append:
        for fidx, frame_rgb in iter_frames(clip, max_frames=max_frames, start_frame=start_frame):
            vid = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            # marcador acumulado (fidx = índice del frame fuente, donde viven los eventos)
            ys = sum(1 for gf, z in goals if gf <= fidx and z == "yellow")
            bs = sum(1 for gf, z in goals if gf <= fidx and z == "blue")
            final_score = (ys, bs)
            # banner activo
            banner = None
            for gf, z in goals:
                if gf <= fidx < gf + banner_frames:
                    banner = (f"GOOOL! Porteria {z}", (fidx - gf) / banner_frames)
                    break
            # métricas del frame
            mdata = {
                "pct_pos": poss.resumen["pct_posesion_total"],
                "pct_ctrl": poss.resumen["pct_control_total"],
                "now_pos": poss.posesion_por_frame.get(fidx),
                "now_ctrl": poss.control_por_frame.get(fidx),
            }
            # feed: últimos max_items con frame_inicio <= fidx
            feed = [(t, c) for fi, t, c in items_all if fi <= fidx][-max_items:][::-1]
            # minimapa / heatmap
            mini = heat = None
            if not degradado:
                renderer.update(mini_by_frame.get(fidx, []))
                mini = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                _accumulate(grid, hm_by_frame.get(fidx, []), bin_cm)
                heat = render_heatmap(grid, bin_cm, sigma_cm=sigma_cm)

            canvas = compose(vid, final_score, mdata, feed, mini, heat, banner, margin_px)
            if sample_frame is not None and sample_png is None and fidx >= sample_frame:
                sample_png = events_paths(tracks_json.stem, "broadcast", "png")
                cv2.imwrite(str(sample_png), canvas)
            append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            n += 1
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()

    resumen = {
        "layout": layout, "goal_source": goal_source, "overlay_degradado": degradado,
        "marcador_final": {"yellow": final_score[0], "blue": final_score[1]},
        "n_frames": n, "fps": fps,
        "conteo": {"goles": len(goals), "eventos_feed": len(items_all)},
    }
    return BroadcastResult(video=out, sample_png=sample_png, resumen=resumen)
