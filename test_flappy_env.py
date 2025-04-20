#Jay Parmar and Matthew Windsor
#CSCI 6660
#test_flappy_env.py: This script tests flappy_env.py by allowing humans to play
#               

#Libraries
from flappy_env import FlappyBirdEnv
import pygame

env = FlappyBirdEnv(render_mode=True)   #Create instance of class
state = env.reset() #Start from first state

running = True  #Set flag for game loop, always running until closed
while running:
    action = 0  #No flap

    #If quit, close game. If spacebar pressed, flap
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            action = 1  #Flap

    #Step environment -> take action, update game by one step
    next_state, reward, done = env.step(action) 
    env.render()    #Render new state

    #If bird dies, reset game
    if done:
        print(f"Game Over! Score: {env.score}")
        state = env.reset()

pygame.quit()

