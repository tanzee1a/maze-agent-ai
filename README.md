# 🧩 Blind Maze Agent

A blind AI agent that learns to navigate a dynamic maze through exploration and memory. The agent starts with zero map knowledge and improves its route across multiple runs using greedy search and A*.

---

## How It Works

### Run 1 — Blind Greedy Exploration
The agent knows only its start position and the goal coordinates. It has no map.

At every new cell it **senses all 4 directions** by querying the maze, then saves which ones are open to a JSON memory file. It picks the next direction using two priorities:

1. **Unexplored first** — bonus for directions never tried before
2. **Closest to goal** — Manhattan distance `|row_diff| + |col_diff|` as a tiebreaker

This is a greedy approach — it only scores the immediate next step, not the full path. The agent will hit dead ends and backtrack. That's expected. The goal of run 1 is to **survive and collect the map**.

### Run 2+ — A* on Known Map
The agent loads its JSON memory from the previous run. It now has a partial or full map of cell exits. It runs **A\*** over that map to find the shortest path.

A\* is smarter than greedy because:
- Greedy only asks: *"how far am I from the goal right now?"*
- A\* asks: *"how far did I travel to get here + how far is the goal?"*

This means A\* always finds the **optimal route** through the known map and never wastes steps chasing dead ends.

If A\* can't find a complete path (undiscovered sections), the agent falls back to greedy exploration to fill in the gaps.

---

## Hazards

The maze contains dynamic hazards that change each turn:

| Icon | Type | Effect |
|------|------|--------|
| 🔥 | Fire | Kills the agent — respawn at start |
| 🟡 | Teleport Yellow | Teleports to paired yellow cell |
| 🟢 | Teleport Green | Teleports to paired green cell |
| 🟣 | Teleport Purple | Teleports to paired purple cell |
| ❄️ | Confusion | Randomizes next move direction |

Fire spreads every **even turn** (map's turn). The agent moves up to **5 steps per odd turn**.

---

## Scoring & Decision Logic

Each direction from a cell is scored as:

```
score = manhattan_distance + (danger * DANGER_WEIGHT) - (unexplored_bonus)

danger          = deaths / (visits + 1)
unexplored_bonus = 10 if visits == 0 else 0
DANGER_WEIGHT   = 5
```

Directions with high death counts are avoided but not permanently banned — since fire moves, a previously deadly path may be safe next run.

---

## Memory Format

All knowledge is persisted to `results/maze_knowledge.json`:

```json
{
  "runs": 3,
  "cell_exits": {
    "(1, 1)": {
      "up":    { "visits": 3, "deaths": 1 },
      "right": { "visits": 2, "deaths": 0 }
    }
  },
  "move_history": [
    { "run": 1, "moves": [[1,2],[1,3],...] }
  ]
}
```

- **visits** — how many times the agent moved in that direction from that cell
- **deaths** — how many times it died doing so
- Knowledge is **never overwritten** — merges use `max()` so no information is lost across runs

---

## Project Structure

```
maze-agent-ai/
├── maze_solver.py       # Agent, World, save/load, render
├── maze_reader.py       # Maze parser, hazard loader, wall logic
├── MAZE_0.png           # Maze walls image
├── MAZE_1.png           # Hazards overlay image
└── results/
    ├── maze_knowledge.json        # Persistent agent memory
    ├── path_run1_blind.png        # Red path — full blind run with step numbers
    └── path_run1_optimized.png    # Blue path — clean shortest route
```

---

## Output Maps

After every run two PNGs are saved to `results/`:

- **`_blind.png`** — Red path showing every step the agent took, including backtracks and deaths. Step numbers are printed on each cell.
- **`_optimized.png`** — Blue path showing only the final route from start to goal. Rendered only on successful runs.

---

## Running

```bash
python maze_solver.py
```

Run it multiple times on the same maze. Each run loads the previous memory and gets faster. To start fresh, delete `results/maze_knowledge.json`.

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `DANGER_WEIGHT` | `5` | How strongly the agent avoids previously deadly directions |
| `REVISIT_WEIGHT` | `3` | Penalty for revisiting already-seen cells |
| `MAX_TURNS` | `10,000` | Hard cutoff to prevent infinite loops |
| `CELL_PX` | `20` | Pixel size per cell in rendered output |