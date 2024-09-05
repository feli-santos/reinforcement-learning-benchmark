import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from evaluate.evaluate_a2c import evaluate_a2c
from evaluate.evaluate_dqn import evaluate_dqn
from evaluate.evaluate_ppo import evaluate_ppo
from scripts.utils import configure_logging, read_json_config
from train.train_a2c import train_a2c
from train.train_dqn import train_dqn
from train.train_ppo import train_ppo

logger = configure_logging()


def run_training(exp, exp_id):
    logger.info(
        f"Starting training for experiment {exp_id} with algorithm {exp['algorithm']}"
    )
    if exp["algorithm"] == "ppo":
        train_ppo(
            exp["environment"],
            exp["total_timesteps"],
            exp["model_save_path"],
            exp["log_path"],
            exp_id,
        )
    elif exp["algorithm"] == "dqn":
        train_dqn(
            exp["environment"],
            exp["total_timesteps"],
            exp["model_save_path"],
            exp["log_path"],
            exp_id,
        )
    elif exp["algorithm"] == "a2c":
        train_a2c(
            exp["environment"],
            exp["total_timesteps"],
            exp["model_save_path"],
            exp["log_path"],
            exp_id,
        )
    logger.info(
        f"Training complete for experiment {exp_id} with algorithm {exp['algorithm']}"
    )


def run_evaluation(exp, exp_id, num_episodes=1):
    logger.info(
        f"Starting evaluation for experiment {exp_id} with algorithm {exp['algorithm']}"
    )
    if exp["algorithm"] == "ppo":
        model_path = os.path.join(exp["model_save_path"], f"ppo_final_{exp_id}")
        evaluate_ppo(exp["environment"], model_path, exp_id, num_episodes)
    elif exp["algorithm"] == "dqn":
        model_path = os.path.join(exp["model_save_path"], f"dqn_final_{exp_id}")
        evaluate_dqn(exp["environment"], model_path, exp_id, num_episodes)
    elif exp["algorithm"] == "a2c":
        model_path = os.path.join(exp["model_save_path"], f"a2c_final_{exp_id}")
        evaluate_a2c(exp["environment"], model_path, exp_id, num_episodes)
    logger.info(
        f"Evaluation complete for experiment {exp_id} with algorithm {exp['algorithm']}"
    )


if __name__ == "__main__":
    config = read_json_config("configs/experiment.json")

    # Run training experiments in parallel
    # with ProcessPoolExecutor(max_workers=len(config["train_experiments"])) as executor:
    #     futures = [
    #         executor.submit(run_training, exp, i)
    #         for i, exp in enumerate(config["train_experiments"])
    #     ]
    #     for future in futures:
    #         try:
    #             future.result()
    #         except Exception as e:
    #             logger.error(f"Training experiment failed with exception: {e}")

    # Run evaluation experiments sequentially to avoid threading issues with graphical libraries
    for i, exp in enumerate(config["evaluate_experiments"]):
        try:
            run_evaluation(exp, i)
        except Exception as e:
            logger.error(f"Evaluation experiment failed with exception: {e}")
