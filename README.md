# Reinforcement Learning Benchmark Project

This project aims to benchmark different reinforcement learning algorithms using various Gymnasium environments. The focus is on comparing the performance of algorithms such as PPO, DQN, and A2C across different tasks.

## Project Structure

```
reinforcement_learning_benchmark/
├── configs/
│   ├── algorithms/
│   │   ├── dqn.json
│   │   ├── ppo.json
│   │   └── a2c.json
│   └── simulation_config.json
├── environments/
│   ├── __init__.py
│   └── lunar_lander.py
├── models/
│   ├── __init__.py
│   ├── dqn.py
│   ├── ppo.py
│   └── a2c.py
├── train/
│   ├── __init__.py
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── train_a2c.py
├── evaluate/
│   ├── __init__.py
│   ├── evaluate_dqn.py
│   ├── evaluate_ppo.py
│   └── evaluate_a2c.py
├── logs/
│   ├── dqn/
│   ├── ppo/
│   └── a2c/
├── results/
│   ├── dqn/
│   ├── ppo/
│   └── a2c/
├── scripts/
│   ├── run_training.py
│   └── run_evaluation.py
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.8.19 (Developed and tested on this version)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
- [Gymnasium](https://gymnasium.farama.org/)

### Project Setup

1. Clone the repository:

```bash
git clone <repository_url>
cd reinforcement_learning_benchmark
```

2. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Benchmarks

#### Training

To train an algorithm on a specific environment, use the corresponding training script. For example, to train DQN on the Lunar Lander environment:

```bash
python scripts/run_training.py --algorithm dqn --env lunar_lander
```

#### Evaluation

To evaluate a trained model, use the corresponding evaluation script. For example, to evaluate a DQN model:

```bash
python scripts/run_evaluation.py --algorithm dqn --env lunar_lander
```

## Configuration Files

- `configs/algorithms/`: Contains JSON files with hyperparameters for each algorithm.
- `configs/simulation_config.json`: Contains general configurations for the simulations, including which environments and algorithms to use.

## Logs and Results

- `logs/`: Contains training logs for each algorithm.
- `results/`: Contains evaluation results for each algorithm.

## Contributing

Feel free to submit issues and pull requests to improve this project.

## License

This project is licensed under the MIT License.

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

