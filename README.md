# FlappyBirdAI

Welcome to our CSCI 6660 Intro to AI final project. This project features a Double Deep-Q Network reinforcement learning agent to play the game Flappy Bird.
Below will explain to you the files, how to play the game yourself, how to train your agent, how to change the rewards and agent's settings, and how to view the different models.

*Important libraries you will need are pygame, pytorch, numpy, and imagerio.

__Files__
- game.py: Game environment for humans, base code for future files (NOT NEEDED FOR AI)
- flappy_env.py: Game environment for AI, includes state observations, actions, and rewards along with game settings
- test_flappy_env.py: File to test environment, flappy_env.py (NOT NEEDED FOR AI)
- agent.py: Reinforcement Learning DQN Flappy Bird Agent
- train.py: Script that trains the agent using flappy_env.py, handles training loop, periodic evaluation, saving trained models
- test_agent.py: Script to run trained models from train.py
- 6660TermProjectData: We added some of our data to show you examples. Each test file includes a .txt file where you can see the settings we used. Tests before Test7 did not include the .gif files.
- 6660FinalSlides: These are the slides used in the video explanation of our project. Link to the video here --> https://youtu.be/lnRS4cJBlA8

__How to Play Yourself__
- There are two ways, one is to simply run game.py, and the other is to run test_lappy_env.py. The only difference between the two is that test_flappy_env.py will show you the rewards as you collect them.

__How to Change Rewards and Settings__
- To change the rewards, open flappy_env.py. Under the step() function, there are 4 rewards you can change. You can experiment with different combinations similar to the project.
- To change the number of layers in the network, open agent.py. Under the DQN class, there is the __init__ function. There are 4 numbers that should all be 256. This is the number of layers.
  You can change the number of layers by halving or doubling the numbers. Ex. 128, 256, 512, 1024 
- To change the agent's settings like gamma, epsilon, or learning rate, open agent.py. Under the DQNAgent class, there is the __init__() function. The parameters include the different variables
  that you can change to your liking.

__How to Train the Agent__
- To train the agent, open train.py. Some settings you can change here are the number of episodes the agent runs for and the checkpoint. The checkpoint dictates how many episodes will pass before
  that model is saved. Then simply run the file and the training will begin.

__How to View Different Models__
- As your agent is training, each best model and checkpoint are saved into the same folder of your project. There will also be a graph where you can see the training summary.
- When saved, there will be a .pth and a .gif file. The .gif file is the actual game in which the model received their score. The .pth file is the model itself, which you can run.
- To run the .pth files, copy the file name that you will want to run. Open test_agent.py and on line 21, paste the file name into the section where it says "test9_bestmodel.pth". Then simply run the file.
