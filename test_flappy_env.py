#Jay Parmar and Matthew Windsor
#CSCI 6660
#test_flappy_env.py: This script tests flappy_env.py by allowing humans to play

#Libraries
from flappy_env import FlappyBirdEnv
import pygame

env = FlappyBirdEnv(render_mode=True)  #Create instance of class
state = env.reset()  #Start from first state

running = True  #Set flag for game loop, always running until closed
total_reward = 0  #Track total reward

while running:
    action = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            action = 1

    next_state, reward, done = env.step(action)
    env.render()

    total_reward += reward  #Add up rewards
    print(f"Reward: {reward}, Score: {env.score}")

    if done:
        print(f"Game Over! Final Score: {env.score}, Final Total Reward: {total_reward}")
        state = env.reset()
        total_reward = 0  #Reset for next game

pygame.quit()
