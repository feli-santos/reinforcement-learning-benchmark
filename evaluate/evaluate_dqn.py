import logging
import os

import gymnasium as gym
from stable_baselines3 import DQN

from scripts.utils import configure_logging, read_json_config

configure_logging()
logger = logging.getLogger(__name__)


def evaluate_dqn(env_name, model_path, exp_id, num_episodes=3):
    logger.info(f"[{exp_id}] Starting DQN evaluation on {env_name}")
    env = gym.make(env_name, render_mode="human")
    model = DQN.load(model_path)

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()

            if terminated or truncated:
                done = True

        logger.info(f"[{exp_id}] Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()  # Ensure the environment is closed after evaluation
    logger.info(f"[{exp_id}] Evaluation complete.")


if __name__ == "__main__":
    config = read_json_config("../configs/experiment_config.json")
    for i, exp in enumerate(config["evaluate_experiments"]):
        if exp["algorithm"] == "dqn":
            model_path = os.path.join(exp["model_save_path"], "dqn_final")
            evaluate_dqn(exp["environment"], model_path, i)
