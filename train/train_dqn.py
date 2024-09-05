import logging
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.utils import configure_logging, read_json_config

configure_logging()
logger = logging.getLogger(__name__)


def train_dqn(env_name, total_timesteps, model_save_path, log_path, exp_id):
    logger.info(f"[{exp_id}] Starting DQN training on {env_name}")
    env = make_vec_env(env_name, n_envs=1, vec_env_cls=DummyVecEnv)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=model_save_path, name_prefix=f"dqn_model_{exp_id}"
    )
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(os.path.join(model_save_path, f"dqn_final_{exp_id}"))
    logger.info(f"[{exp_id}] Training complete and model saved.")


if __name__ == "__main__":
    config = read_json_config("../configs/experiment_config.json")
    for i, exp in enumerate(config["train_experiments"]):
        if exp["algorithm"] == "dqn":
            train_dqn(
                exp["environment"],
                exp["total_timesteps"],
                exp["model_save_path"],
                exp["log_path"],
                i,
            )
