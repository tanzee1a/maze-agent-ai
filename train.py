import os
import argparse
import numpy as np
import time
from agent import HybridAgent
from environment import MazeEnvironment
from visualizer import MazeVisualizer

# ── Argument Parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Silent Cartographer — Q-Learning Training")
parser.add_argument(
    "--maze", "-m",
    choices=["alpha", "beta", "gamma"],
    default="alpha",
    help="Which maze to run (default: alpha)"
)
parser.add_argument(
    "--episodes", "-e",
    type=int,
    default=5,
    help="Number of TRAINING episodes for maze-alpha (default: 5, ignored for beta/gamma)"
)
parser.add_argument(
    "--gif-skip", "-g",
    type=int,
    default=200,
    help="Capture every Nth step as a GIF frame (default: 200, higher = smaller file)"
)
parser.add_argument(
    "--gif-fps", "-f",
    type=int,
    default=12,
    help="GIF playback speed in frames per second (default: 12)"
)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
MAZE_NAME      = args.maze
MAZE_DIR       = os.path.join("TestMazes", f"maze-{MAZE_NAME}")
MAZE_PATH      = os.path.join(MAZE_DIR, "MAZE_0.png")
HAZARD_PATH    = os.path.join(MAZE_DIR, "MAZE_1.png")

RUN_DIR        = os.path.join("runs", MAZE_NAME)
VIZ_DIR        = os.path.join(RUN_DIR, "viz")
SAVE_PATH      = os.path.join(RUN_DIR, "q_table.npy")

TEST_EPISODES  = 5
MAX_TURNS      = 10_000


def run_episodes(env, agent, viz, num_episodes, mode, start_time, episode_offset=0, freeze_epsilon=False):
    assert mode in ("train", "test")

    saved_epsilon = None
    if freeze_epsilon:
        saved_epsilon = agent.epsilon
        agent.epsilon = 0.0

    results = []

    for ep in range(1, num_episodes + 1):
        episode_num = episode_offset + ep

        start_pos   = env.reset()
        agent.reset_episode(start_pos)
        if freeze_epsilon:
            agent.epsilon = 0.0

        last_result = None
        success     = False
        path_taken  = [start_pos]
        death_cells = []
        step_count  = 0

        for turn in range(MAX_TURNS):
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)

            path_taken.append(last_result.current_position)
            step_count += 1

            if last_result.is_dead:
                death_cells.append(last_result.current_position)
                agent.current_pos = env.start

            # ── Capture frame for GIF (every step, visualizer applies skip) ──
            viz.capture_frame(
                agent        = agent,
                env          = env,
                path_so_far  = path_taken,
                deaths_so_far= death_cells,
                episode_num  = episode_num,
            )

            if last_result.is_goal_reached:
                agent._process_result(last_result)
                success = True
                agent.all_path_lengths[-1] = env.get_episode_stats()["path_length"]
                print(f"  [DEBUG] Goal reached! turn={turn} pos={last_result.current_position} agent.goal_pos={agent.goal_pos}")
                break

        if not success and (last_result is None or not last_result.is_goal_reached):
            agent.finish_episode_timeout()

        stats   = env.get_episode_stats()
        m       = agent.get_metrics()
        elapsed = time.time() - start_time
        status  = "SUCCESS" if success else "timeout"
        tag     = f"[{mode.upper()}]"

        print(
            f"{tag} ep {episode_num:>4} | {status} | "
            f"turns={stats['turns_taken']:>6} | "
            f"deaths={stats['deaths']:>2} | "
            f"cells={m['unique_cells']:>4} | "
            f"goal={'YES' if m['goal_found'] else 'no':<3} | "
            f"e={m['epsilon']:.3f} | "
            f"success={m['success_rate']:>5.1f}% | "
            f"t={elapsed:.0f}s"
        )

        # ── Save JPEG + GIF, clear frame buffer ───────────────────────────────
        viz.save_episode(
            episode_num = episode_num,
            agent       = agent,
            env         = env,
            path_taken  = path_taken,
            death_cells = death_cells,
            output_dir  = os.path.join(VIZ_DIR, mode),
        )

        results.append({
            "episode":     episode_num,
            "success":     success,
            "turns":       stats["turns_taken"],
            "deaths":      stats["deaths"],
            "path_length": stats.get("path_length", None) if success else None,
        })

    if saved_epsilon is not None:
        agent.epsilon = saved_epsilon

    return results


def print_report(results, label):
    print()
    print("=" * 55)
    print(f"{label} RESULTS")
    print("=" * 55)
    successes    = [r for r in results if r["success"]]
    success_rate = len(successes) / len(results) * 100
    path_lengths = [r["path_length"] for r in successes if r["path_length"] is not None]
    avg_path     = np.mean(path_lengths) if path_lengths else float("nan")
    avg_turns    = np.mean([r["turns"]  for r in results])
    avg_deaths   = np.mean([r["deaths"] for r in results])

    print(f"  Episodes:          {len(results)}")
    print(f"  Successes:         {len(successes)} / {len(results)}")
    print(f"  Success rate:      {success_rate:.1f}%")
    print(f"  Avg path length:   {avg_path:.1f}  (successful episodes only)")
    print(f"  Avg turns/episode: {avg_turns:.1f}")
    print(f"  Avg deaths/ep:     {avg_deaths:.1f}")
    print("=" * 55)


def main():
    print("=" * 55)
    print(f"SILENT CARTOGRAPHER  [{MAZE_NAME.upper()}]")
    print("=" * 55)
    print(f"Maze:   {MAZE_PATH}")
    print(f"Hazard: {HAZARD_PATH}")
    print(f"Viz:    {VIZ_DIR}/")
    print(f"Q-save: {SAVE_PATH}")
    print(f"GIF:    skip={args.gif_skip}  fps={args.gif_fps}")
    print()

    if MAZE_NAME == "alpha":
        os.makedirs(os.path.join(VIZ_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(VIZ_DIR, "test"), exist_ok=True)

    env   = MazeEnvironment(MAZE_PATH, HAZARD_PATH)
    agent = HybridAgent()
    viz   = MazeVisualizer(MAZE_PATH, gif_fps=args.gif_fps, gif_skip=args.gif_skip)

    if MAZE_NAME == "alpha" and os.path.exists(SAVE_PATH):
        agent.q_table = np.load(SAVE_PATH)
        print(f"Loaded Q-table from {SAVE_PATH}  (non-zero entries: {np.count_nonzero(agent.q_table)})")
    else:
        print("Starting fresh Q-table.")

    start_time = time.time()

    if MAZE_NAME == "alpha":
        num_train = args.episodes
        print(f"\n-- TRAINING  ({num_train} episodes) --")
        train_results = run_episodes(
            env, agent, viz, num_train,
            mode="train", start_time=start_time
        )
        np.save(SAVE_PATH, agent.q_table)
        print(f"\nQ-table saved -> {SAVE_PATH}")
        print_report(train_results, "TRAINING")

    is_alpha  = MAZE_NAME == "alpha"
    train_eps = args.episodes if is_alpha else 0
    eps_note  = "e=0 exploit" if is_alpha else "e active, exploring"

    print(f"\n-- TESTING  ({TEST_EPISODES} episodes, {eps_note}) --")
    test_results = run_episodes(
        env, agent, viz, TEST_EPISODES,
        mode="test", start_time=start_time,
        episode_offset=train_eps, freeze_epsilon=is_alpha
    )

    print_report(test_results, "TEST")
    print(f"\nEpisode images -> {VIZ_DIR}/")


if __name__ == "__main__":
    main()
