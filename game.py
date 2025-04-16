#Matthew Windsor & Jay Parmar
#CSCI 6660
#game.py: Mimic of Flappy Bird for people. One change includes a red item that can be collected for points. 
#         +1 point for time alive, +5 points for passing through a pipe, +10 for collecting an item


#Libraries
import pygame
import sys
import random

#Initialize Pygame
pygame.init()

#Game settings
WIDTH, HEIGHT = 400, 600
FPS = 60
GRAVITY = 0.5
JUMP_STRENGTH = -10

#Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird Clone")
clock = pygame.time.Clock()

#Colors/Game Font
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
font = pygame.font.SysFont(None, 36)

#Bird Entity
bird_x = 50
bird_y = HEIGHT // 2
bird_velocity = 0
bird_radius = 20

#Pipe Entity
pipe_width = 70
pipe_gap = 150
pipe_speed = 3
pipe_spacing = 250

#Item Entity
item_radius = 10

#Score Counter
score = 0
frame_counter = 0

#Create a pipe function
def create_pipe(x_pos, with_item=False):    #(x-cord of where pipe will appear, boolean if pipe has item)
    top = random.randint(50, HEIGHT - pipe_gap - 50)    #Random length of top pipe 
    bottom = top + pipe_gap                             #Calculate length of bottom pipe
    item = None
    if with_item:   #If there is an item
        item_x = x_pos + pipe_width // 2    #Places item inbetween pipe x-axis
        item_y = (top + bottom) // 2        #Places item inbetween pipe y-axis
        item = {'x': item_x, 'y': item_y, 'active': True}   #Creates item object with cords and collection status

    return {'x': x_pos, 'top': top, 'bottom': bottom, 'item': item, 'passed': False}

#Initialize pipes
pipes = []  #Empty list of pipes
for i in range(3):  #Creates 3 sets of pipes 
    with_item = (i == 1)
    pipes.append(create_pipe(WIDTH + i * pipe_spacing, with_item=with_item))

#Game loop
running = True
while running:

    #Frame settings
    clock.tick(FPS)
    frame_counter += 1

    #Events for quitting the game and a keyboard press
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            bird_velocity = JUMP_STRENGTH

    #Add +1 score every second
    if frame_counter >= FPS:
        score += 1
        frame_counter = 0

    #Update bird
    bird_velocity += GRAVITY
    bird_y += bird_velocity

    #Move pipes and items
    for pipe in pipes:  #Iterate through list of pipes
        pipe['x'] -= pipe_speed     #Moves pipe to left by pipe_speed variable
        if pipe['item'] and pipe['item']['active']: #If there is a pipe with an item uncollected
            pipe['item']['x'] -= pipe_speed         #Move item with pipe

    #Recycle pipes
    for i in range(len(pipes)):
        if pipes[i]['x'] + pipe_width < 0:  #Check if pipe is offscreen
            farthest_x = max(p['x'] for p in pipes) #Finds x cord of leftmost pipe
            with_item = random.random() < 0.5   #Determines if new pipe has item, if rand_num < .5
            pipes[i] = create_pipe(farthest_x + pipe_spacing, with_item=with_item)  #Creates new pipe using function

    #Score +5 when passing a pipe
    for pipe in pipes:
        if not pipe['passed'] and pipe['x'] + pipe_width < bird_x:
            score += 5
            pipe['passed'] = True

    #Collision for screen edges
    if bird_y - bird_radius < 0 or bird_y + bird_radius > HEIGHT:
        print("Game Over! Final Score:", score)
        running = False

    #Collision for pipes
    for pipe in pipes:
        if (
            pipe['x'] < bird_x + bird_radius < pipe['x'] + pipe_width and
            (bird_y - bird_radius < pipe['top'] or bird_y + bird_radius > pipe['bottom'])
        ):
            print("Game Over! (Pipe collision) Final Score:", score)
            running = False

    #Collision for collectibles
    for pipe in pipes:
        if pipe['item'] and pipe['item']['active']: #If pipe has active item
            dx = bird_x - pipe['item']['x'] #Difference between bird cord and item cord
            dy = bird_y - pipe['item']['y'] 
            distance = (dx ** 2 + dy ** 2) ** 0.5   #Straight line distance to item 
            if distance < bird_radius + item_radius:   #If distance < sum of their radius (overlapping), update score and remove item
                score += 10
                pipe['item']['active'] = False

    #Draw everything   
    screen.fill(WHITE)  #Background
    pygame.draw.circle(screen, BLACK, (bird_x, int(bird_y)), bird_radius)   #Bird

    #Pipes & Item
    for pipe in pipes:
        pygame.draw.rect(screen, GREEN, (pipe['x'], 0, pipe_width, pipe['top']))
        pygame.draw.rect(screen, GREEN, (pipe['x'], pipe['bottom'], pipe_width, HEIGHT - pipe['bottom']))
        if pipe['item'] and pipe['item']['active']:
            pygame.draw.circle(screen, RED, (int(pipe['item']['x']), int(pipe['item']['y'])), item_radius)

    #Score
    score_surface = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_surface, (10, 10))

    pygame.display.flip()

pygame.quit()
sys.exit()
