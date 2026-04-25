import os
import numpy as np
import time

from agent import HybridAgent
from environment import MazeEnvironment
from visualizer import MazeVisualizer


# ── SELECT MAZE HERE ───────────────────────────────
MAZE_NAME = "beta"   # change: "alpha" / "beta" / "gamma"

EPISODES       = 5
TEST_EPISODES  = 5
MAX_TURNS      = 10_000
GIF_SKIP       = 200
GIF_FPS        = 12


# ── PATHS ──────────────────────────────────────────
MAZE_DIR  = os.path.join("TestMazes", f"maze-{MAZE_NAME}")
MAZE_PATH = os.path.join(MAZE_DIR, "MAZE_0.png")

RUN_DIR   = os.path.join("runs", MAZE_NAME)
VIZ_DIR   = os.path.join(RUN_DIR, "viz")
SAVE_PATH = os.path.join(RUN_DIR, "q_table.npy")


# ───────────────────────────────────────────────────
# RUN EPISODES
# ───────────────────────────────────────────────────
def run_episodes(env, agent, viz, num_episodes, mode, start_time):

    results = []

    for ep in range(1, num_episodes + 1):

        start_pos = env.reset()
        agent.reset_episode(start_pos)

        last_result = None
        success = False

        path_taken = [start_pos]
        death_cells = []
        current_segment = [start_pos]

        for turn in range(MAX_TURNS):

            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)

            path_taken.append(last_result.current_position)
            current_segment.append(last_result.current_position)

            if last_result.is_dead:
                death_cells.append(last_result.current_position)
                agent.current_pos = env.start
                current_segment = [env.start]

            # ── CAPTURE FRAME ───────────────────────
            viz.capture_frame(
                agent=agent,
                env=env,
                path_so_far=current_segment,
                deaths_so_far=death_cells,
                episode_num=ep,
            )

            if last_result.is_goal_reached:
                agent._process_result(last_result)
                success = True
                break

        if not success:
            agent.finish_episode_timeout()

        stats = env.get_episode_stats()
        m = agent.get_metrics()

        print(
            f"[{mode.upper()}] ep {ep:>3} | "
            f"{'SUCCESS' if success else 'timeout'} | "
            f"turns={stats['turns_taken']:>5} | "
            f"deaths={stats['deaths']:>2} | "
            f"cells={m['unique_cells']:>4} | "
            f"goal={'YES' if m['goal_found'] else 'no':<3} | "
            f"e={m['epsilon']:.3f}"
        )

        # ── SAVE GIF + PNG ────────────────────────
        viz.save_episode(
            episode_num=ep,
            agent=agent,
            env=env,
            path_taken=path_taken,
            death_cells=death_cells,
            output_dir=os.path.join(VIZ_DIR, mode),
        )

        results.append({
            "episode":     ep,
            "success":     success,
            "turns":       stats["turns_taken"],
            "deaths":      stats["deaths"],
            "path_length": stats["path_length"] if success else None,  # cells visited, no teleport jumps
            "cells_explored": stats["cells_explored"],  # unique cells this episode
            "cells_visited":  stats["path_length"],        # total visits this episode (incl. revisits)
        })

    return results


# ───────────────────────────────────────────────────
# REPORT
# ───────────────────────────────────────────────────
def print_report(results, label, agent=None):
    print("\n" + "=" * 50)
    print(label)
    print("=" * 50)

    successes    = [r for r in results if r["success"]]
    success_rate = len(successes) / len(results) * 100

    # avg path length and avg turns: successful episodes only (per spec)
    path_lengths = [r["path_length"] for r in successes if r["path_length"] is not None]
    succ_turns   = [r["turns"] for r in successes]
    avg_path     = np.mean(path_lengths) if path_lengths else float("nan")
    avg_turns    = np.mean(succ_turns)   if succ_turns   else float("nan")

    # death rate: total_deaths / total_turns across all episodes (per spec)
    total_deaths = sum(r["deaths"] for r in results)
    total_turns  = sum(r["turns"]  for r in results)
    death_rate   = total_deaths / total_turns if total_turns > 0 else float("nan")

    print(f"Episodes:       {len(results)}")
    print(f"Success rate:   {success_rate:.1f}%")
    print(f"Avg path len:   {avg_path:.1f}  (successful only)")
    print(f"Avg turns:      {avg_turns:.1f}  (successful only)")
    print(f"Death rate:     {death_rate:.5f}  (total_deaths / total_turns)")

    if agent is not None:
        # Exploration efficiency: unique cells seen / total cell visits across these episodes
        # Both from env stats so they cover the same scope. Max 1.0 = zero revisits.
        total_cell_visits  = sum(r["cells_visited"]  for r in results)
        total_cells_unique = sum(r["cells_explored"] for r in results)
        expl_eff           = total_cells_unique / total_cell_visits if total_cell_visits > 0 else float("nan")

        # Map completeness: fraction of full grid discovered
        map_complete = agent.get_metrics()["map_completeness"]

        # Learning efficiency: which episode the goal was first found
        first_success = next((r["episode"] for r in results if r["success"]), None)

        print(f"Exploration eff:  {expl_eff:.3f}  (unique / total cell visits)")
        print(f"Map completeness: {map_complete * 100:.1f}%")
        if first_success:
            print(f"First success:    episode {first_success}")
        else:
            print(f"First success:    never")

    print("=" * 50)


# ───────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────
def main():
    print(f"\n=== RUNNING MAZE: {MAZE_NAME.upper()} ===\n")

    os.makedirs(os.path.join(VIZ_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(VIZ_DIR, "test"), exist_ok=True)

    env = MazeEnvironment(MAZE_NAME)
    agent = HybridAgent()

    viz = MazeVisualizer(
        MAZE_PATH,
        gif_fps=GIF_FPS,
        gif_skip=GIF_SKIP
    )

    start_time = time.time()

    # ── ALPHA: TRAIN + TEST ───────────────────────
    if MAZE_NAME == "alpha":

        # ── Load Q-table only for alpha ───────────
        if os.path.exists(SAVE_PATH):
            agent.q_table = np.load(SAVE_PATH)
            print("Loaded existing alpha Q-table")
        else:
            print("Starting fresh alpha Q-table")

        print("\n--- TRAIN ---\n")

        train_results = run_episodes(
            env, agent, viz,
            EPISODES,
            mode="train",
            start_time=start_time
        )

        np.save(SAVE_PATH, agent.q_table)
        print_report(train_results, "TRAIN RESULTS", agent=agent)

        print("\n--- TEST ---\n")

        saved_epsilon = agent.epsilon
        agent.epsilon = 0.0  # deterministic test only for alpha

        test_results = run_episodes(
            env, agent, viz,
            TEST_EPISODES,
            mode="test",
            start_time=start_time
        )

        agent.epsilon = saved_epsilon

        print_report(test_results,  "TEST RESULTS",  agent=agent)

    # ── BETA/GAMMA: TEST ONLY, BUT STILL EXPLORE ──
    else:
        print(f"Testing {MAZE_NAME} with a fresh agent state")
        print("\n--- TEST ---\n")

        test_results = run_episodes(
            env, agent, viz,
            EPISODES,
            mode="test",
            start_time=start_time
        )

        print_report(test_results,  "TEST RESULTS",  agent=agent)

    print("\nDone. Check runs/ folder.\n")


# ───────────────────────────────────────────────────
if __name__ == "__main__":
    main()
