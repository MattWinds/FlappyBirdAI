# FlappyBirdAI

Files
- game.py: Game environment for humans, base code for future files (NOT NEEDED FOR AI)
- flappy_env.py: Game environment for AI, includes state observations, actions, and rewards along with game settings
- test_flappy_env.py: File to test environment, flappy_env.py (NOT NEEDED FOR AI)
- agent.py: Reinforcement Learning DQN Flappy Bird Agent
- train.py: Script that trains the agent using flappy_env.py, handles training loop, periodic evaluation, saving trained models
- test_agent.py: Script to run trained models from train.py
