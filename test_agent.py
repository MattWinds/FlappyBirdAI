#Matthew Windsor and Jay Parmar
#CSCI 6660
#test_agent.py: Load a trained DQN agent (model) and test it playing flappy bird without exploration

#Libraries
import torch
from flappy_env import FlappyBirdEnv
from agent import DQNAgent

#Create the game environment
env = FlappyBirdEnv(render_mode=True)

# Set up agent
state_size = len(env._get_state())  #Size of the state vector, should be 6
action_size = 2 #Possible actions: 0 = no flap, 1 = flap

#Create DQN agent
agent = DQNAgent(state_size, action_size)

#Load best model (adjust filename as needed)
agent.model.load_state_dict(torch.load("dqn_flappy_checkpoint_16000.pth", map_location=torch.device('cpu')))
agent.model.eval()  #Evaluation mode (no dropout, no randomness)

#Set no exploration during test (pure exploitation, no random actions)
agent.epsilon = 0.0

#Test loop
state = env.reset() #Reset state
done = False
total_reward = 0    #Track total reward earned during this test run

#While the bird is alive
while not done:
    action = agent.act(state)   #Agent chooses the action (purely based on learned policy)
    next_state, reward, done = env.step(action) #Apply action to the environment, get new state and reward
    env.render()     #Render the environment visually
    state = next_state  #Move to the next state
    total_reward += reward   #Accumulate total reward

print(f"Test completed! Total Reward: {total_reward}")
