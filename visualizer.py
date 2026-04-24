"""
visualizer.py
=============
Draws the agent's episode onto the maze image and saves a JPEG + animated GIF.

Per-episode JPEG shows:
  - Visited cells (light green)
  - Hazards: fire (red-orange), push-up (steel blue), push-left (cornflower)
  - Confusion traps (purple)
  - Teleporters (orange)
  - Path taken (blue line)
  - Death locations (red X)
  - Start (cyan dot), Goal (gold dot), Agent final pos (white dot)
  - Legend + stats strip

Per-episode GIF shows:
  - All of the above, animated step-by-step as the agent moves

Usage:
    viz = MazeVisualizer("MAZE_0.png")

    # During the episode — call after every step:
    viz.capture_frame(agent, env, path_so_far, deaths_so_far)

    # After the episode:
    viz.save_episode(episode_num, agent, env, path_taken, death_cells, output_dir)
    # ↑ saves both episode_NNN.jpg and episode_NNN.gif, then clears frame buffer
"""

import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ── Constants — must match maze_reader.py ────────────────────────────────────
GRID  = 64
WALL  = 2
STEP  = 16
INNER = 14

# ── Colours ───────────────────────────────────────────────────────────────────
C_VISITED    = (144, 238, 144, 80)
C_FIRE       = (255,  80,   0, 180)
C_PUSH_UP    = ( 70, 130, 180, 150)
C_PUSH_LEFT  = (100, 149, 237, 150)
C_CONFUSION  = (180,   0, 255, 150)
C_TELEPORT   = (255, 165,   0, 170)
C_PATH       = ( 30, 144, 255, 220)
C_DEATH_LINE = (220,   0,   0)
C_START      = (  0, 255, 255)
C_GOAL       = (255, 215,   0)
C_AGENT      = (255, 255, 255)

LEGEND_ITEMS = [
    (C_VISITED[:3],   "visited"),
    (C_PATH[:3],      "path"),
    (C_FIRE[:3],      "fire"),
    (C_DEATH_LINE,    "death"),
    (C_START,         "start"),
    (C_GOAL,          "goal"),
    (C_CONFUSION[:3], "confusion"),
    (C_TELEPORT[:3],  "teleport"),
    (C_PUSH_UP[:3],   "push-up"),
    (C_PUSH_LEFT[:3], "push-left"),
]


# ── Pixel helpers ─────────────────────────────────────────────────────────────

def cell_to_pixel(row: int, col: int) -> Tuple[int, int]:
    return WALL + col * STEP, WALL + row * STEP


def cell_center_px(row: int, col: int) -> Tuple[int, int]:
    return WALL + col * STEP + STEP // 2, WALL + row * STEP + STEP // 2


# ── Main class ────────────────────────────────────────────────────────────────

class MazeVisualizer:
    def __init__(self, maze_path: str, gif_fps: int = 12, gif_skip: int = 3):
        """
        Parameters
        ----------
        maze_path : path to MAZE_0.png
        gif_fps   : playback speed of the output GIF
        gif_skip  : only keep every Nth frame (reduces GIF file size)
        """
        self.maze_path  = maze_path
        self.gif_fps    = gif_fps
        self.gif_skip   = gif_skip
        self.base_image = Image.open(maze_path).convert("RGB")
        self._frames: List[Image.Image] = []   # RGB PIL images, one per captured step
        self._step_counter: int = 0

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _fresh_canvas(self) -> Image.Image:
        return self.base_image.copy()

    # ── Low-level drawing ─────────────────────────────────────────────────────

    def _render(
        self,
        agent,
        env,
        path_so_far:   List[Tuple[int, int]],
        deaths_so_far: List[Tuple[int, int]],
        episode_num:   int,
    ) -> Image.Image:
        """
        Render a single frame onto a fresh canvas and return it (RGB Image).
        Called both during capture and for the final JPEG.
        """
        from maze_reader import Hazard

        img  = self._fresh_canvas()
        draw = ImageDraw.Draw(img, "RGBA")

        def fill_cell(r, c, color):
            x, y = cell_to_pixel(r, c)
            draw.rectangle([x + 1, y + 1, x + INNER, y + INNER], fill=color)

        # 1. Visited cells
        for (r, c) in agent.visited:
            fill_cell(r, c, C_VISITED)

        # 2. Hazards
        for (r, c), hz in env.hazards.items():
            if hz == Hazard.FIRE:
                fill_cell(r, c, C_FIRE)
            elif hz == Hazard.PUSH_UP:
                fill_cell(r, c, C_PUSH_UP)
            elif hz == Hazard.PUSH_LEFT:
                fill_cell(r, c, C_PUSH_LEFT)

        # 3. Confusion cells
        for (r, c) in agent.confuse:
            fill_cell(r, c, C_CONFUSION)

        # 4. Teleporter cells
        for (r, c) in agent.teleports:
            fill_cell(r, c, C_TELEPORT)

        # 5. Path line
        if len(path_so_far) >= 2:
            pixels = [cell_center_px(r, c) for r, c in path_so_far]
            for i in range(len(pixels) - 1):
                draw.line([pixels[i], pixels[i + 1]], fill=C_PATH, width=2)

        # 6. Death markers
        for (r, c) in deaths_so_far:
            cx, cy = cell_center_px(r, c)
            s = 4
            draw.line([(cx-s, cy-s), (cx+s, cy+s)], fill=C_DEATH_LINE, width=2)
            draw.line([(cx+s, cy-s), (cx-s, cy+s)], fill=C_DEATH_LINE, width=2)

        # 7. Start
        sr, sc = env.start
        cx, cy = cell_center_px(sr, sc)
        draw.ellipse([(cx-4, cy-4), (cx+4, cy+4)], fill=C_START)

        # 8. Goal
        gr, gc = env.goal
        cx, cy = cell_center_px(gr, gc)
        draw.ellipse([(cx-5, cy-5), (cx+5, cy+5)], fill=C_GOAL)

        # 9. Agent current position
        if agent.current_pos:
            ar, ac = agent.current_pos
            cx, cy = cell_center_px(ar, ac)
            draw.ellipse([(cx-3, cy-3), (cx+3, cy+3)], fill=C_AGENT)

        # 10. Legend
        lx, ly = 4, 4
        lh = len(LEGEND_ITEMS) * 10 + 6
        draw.rectangle([lx, ly, lx + 74, ly + lh], fill=(0, 0, 0, 170))
        for i, (color, label) in enumerate(LEGEND_ITEMS):
            y = ly + 4 + i * 10
            draw.rectangle([lx + 2, y, lx + 8, y + 7], fill=color)
            draw.text((lx + 11, y), label, fill=(255, 255, 255))

        # 11. Bottom stats strip
        m          = agent.get_metrics()
        stats_text = (
            f"Ep {episode_num}  |  "
            f"cells={len(agent.visited)}  |  "
            f"deaths={len(deaths_so_far)}  |  "
            f"goal={'FOUND' if agent.goal_pos else 'not found'}  |  "
            f"success={m['success_rate']}%  |  "
            f"phase={'BFS' if getattr(agent, 'phase', 1) == 1 else 'A*'}"
        )
        img_w, img_h = img.size
        draw.rectangle([0, img_h - 14, img_w, img_h], fill=(0, 0, 0, 210))
        draw.text((4, img_h - 12), stats_text, fill=(255, 255, 255))

        return img.convert("RGB")

    # ── Public API ────────────────────────────────────────────────────────────

    def capture_frame(
        self,
        agent,
        env,
        path_so_far:   List[Tuple[int, int]],
        deaths_so_far: List[Tuple[int, int]],
        episode_num:   int,
    ):
        """
        Call after every step during an episode to build the GIF frame buffer.
        Applies gif_skip to keep file sizes manageable.
        """
        self._step_counter += 1
        if self._step_counter % self.gif_skip != 0:
            return
        frame = self._render(agent, env, path_so_far, deaths_so_far, episode_num)
        self._frames.append(frame)

    def save_episode(
        self,
        episode_num:  int,
        agent,
        env,
        path_taken:   List[Tuple[int, int]],
        death_cells:  List[Tuple[int, int]],
        output_dir:   str = ".",
    ):
        """
        Save a final JPEG and an animated GIF for the episode,
        then clear the frame buffer.

        Returns (jpeg_path, gif_path).
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── Final JPEG (full episode, best quality) ───────────────────────────
        final = self._render(agent, env, path_taken, death_cells, episode_num)
        jpeg_path = os.path.join(output_dir, f"episode_{episode_num:03d}.jpg")
        final.save(jpeg_path, "JPEG", quality=92)
        print(f"  [VIZ] JPEG  → {jpeg_path}")

        # ── Animated GIF ──────────────────────────────────────────────────────
        gif_path = None
        real_frames = [f for f in self._frames if f is not None]

        if real_frames:
            # Always append the final frame so the GIF ends on the full picture
            real_frames.append(final)

            duration_ms = max(1, round(1000 / self.gif_fps))

            gif_path = os.path.join(output_dir, f"episode_{episode_num:03d}.gif")
            real_frames[0].save(
                gif_path,
                format="GIF",
                save_all=True,
                append_images=real_frames[1:],
                duration=duration_ms,
                loop=0,          # loop forever
                optimize=False,
            )
            print(f"  [VIZ] GIF   → {gif_path}  ({len(real_frames)} frames @ {self.gif_fps} fps)")
        else:
            print("  [VIZ] No GIF frames captured (was capture_frame called during the episode?)")

        # Clear buffer for next episode
        self._frames = []
        self._step_counter = 0

        return jpeg_path, gif_path
