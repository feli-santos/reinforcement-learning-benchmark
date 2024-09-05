import gymnasium as gym


def run_random_agent(env_name, num_episodes):
    """Run a random agent on the specified environment for a number of episodes."""
    env = gym.make(env_name, render_mode="human")
    total_rewards = list()

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()

            if terminated:
                done = True

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")


if __name__ == "__main__":
    run_random_agent(env_name="LunarLander-v2", num_episodes=50)
