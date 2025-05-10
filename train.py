#Matthew Windsor and Jay Parmar
#CSCI 6660
#train.py: Training script to train a DQN agent. Creates the environment using flappy_env.py, creates a DQN agent using agent.py.
#           Starts an episode loop, picks actions and tracks rewards and epsilon for data analysis. Saves the best model whenever
#           an episode's total reward is the new max as well as the model every (started at 100 but moved to 1000) episodes.
#           After training is done, reports training time and best reward achieved, and plots two graphs for data analysis.


#Libraries
from flappy_env import FlappyBirdEnv
from agent import DQNAgent
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import imageio

#Hyperparameters
EPISODES = 5000    #Total number of episodes to train
MAX_STEPS = 10000   #Max steps allowed per episode (safety cutoff)
CHECKPOINT = 1000    #How many episodes will pass before a model is saved

#Create game environment
env = FlappyBirdEnv(render_mode=False)
state_size = len(env._get_state())  #Size of the state vector, should be 6
action_size = 2  #Possible actions: 0 = no flap, 1 = flap

#Create DQN agent
agent = DQNAgent(state_size, action_size)

#Create lists for tracking
episode_rewards = []    #To store total reward per episode
episode_epsilons = []   #To store epsilon value per episode
best_reward = -(math.inf) #Initialize best reward tracker

#Start timer for training
start_time = time.time()

#--------------------------------------------------------------------------------------------------------------------------

#Training loop
for episode in range(EPISODES):
    state = env.reset() #Reset the environment at start of episode
    total_reward = 0    #Track total reward this episode

    for step in range(MAX_STEPS):
        action = agent.act(state)   #Agent picks an action
        #Apply action to environment, get next state and reward
        next_state, reward, done = env.step(action) 
        agent.remember(state, action, reward, next_state, done) #Store the experience
        agent.replay()  #Learn from memory

        state = next_state  #Move to the next state
        total_reward += reward  #Accumulate total reward

        #If bird dies, end episode
        if done:
            break
    
    #Track episode results, append to lists
    episode_rewards.append(total_reward)
    episode_epsilons.append(agent.epsilon)

    #Save best model
    if total_reward > best_reward:
        best_reward = total_reward
        model_path = f"dqn_flappy_best_{int(best_reward)}.pth"
        torch.save(agent.model.state_dict(), model_path)
        print(f"✅ New best model saved at episode {episode + 1} with reward {total_reward:.2f}")

        #Record the best episode using current model
        record_env = FlappyBirdEnv(render_mode=True)  #Create a fresh env to replay the episode visually
        state = record_env.reset()
        done = False
        frames = []  #Store frames for GIF

        while not done:
            action = agent.act(state)
            next_state, reward, done = record_env.step(action)

            #Capture the screen frame as an RGB array for GIF generation
            frame = record_env.render(return_rgb_array=True)
            frames.append(frame)
            state = next_state

        #Save the recorded frames as a GIF named by score
        imageio.mimsave(f"best_run_{int(best_reward)}.gif", frames, fps=30)
        print(f"🎥 Best run recorded as best_run_{int(best_reward)}.gif")

    #Save checkpoint every x episodes
    if (episode + 1) % CHECKPOINT == 0:
        torch.save(agent.model.state_dict(), f"dqn_flappy_checkpoint_{episode+1}.pth")

    #Print progress
    print(f"Episode {episode + 1}/{EPISODES} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

#--------------------------------------------------------------------------------------------------------------------------

#After training, time ends
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(int(elapsed_time), 60)
print(f"\nTraining completed in {minutes} minutes and {seconds} seconds.")
print(f"Best reward achieved during training: {best_reward:.2f}")

#Plot rewards
plt.figure(figsize=(12,6))

#Subplot 1: Reward curves
plt.subplot(2,1,1)
plt.plot(episode_rewards, label='Episode Reward')

#Smoothed reward curve
if len(episode_rewards) >= 10:
    moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
    plt.plot(range(9, len(moving_avg)+9), moving_avg, label='Smoothed Reward (10-ep MA)', color='red')

#Plot settings
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress (Reward)')
plt.legend()
plt.grid()

#Subplot 2: Epsilon curve
plt.subplot(2,1,2)
plt.plot(episode_epsilons, label='Epsilon Value', color='purple')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Exploration Decay (Epsilon)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("training_summary.png")
plt.show()
