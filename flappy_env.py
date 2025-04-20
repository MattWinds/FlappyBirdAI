#Jay Parmar and Matthew Windsor
#CSCI 6660
#flappy_env.py: Flappy Bird environment. Defines the settings, pipe generation, collectible item logic, rendering, and reward structure. 
#               This environment serves as the interface between the game and the RL agent, providing state observations and receiving actions.

#Libraries
import pygame
import random
import numpy as np

#Custom environment class
class FlappyBirdEnv:

    #Constructor to initialize settings
    def __init__(self, render_mode=False):
        pygame.init()

        #Game Settings
        self.WIDTH, self.HEIGHT = 400, 600
        self.FPS = 60
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -10
        self.pipe_gap = 150
        self.pipe_speed = 3
        self.pipe_width = 70
        self.pipe_spacing = 250
        self.bird_radius = 20
        self.bird_x = 50
        self.item_radius = 10

        #Render Settings
        self.render_mode = render_mode
        if render_mode:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Flappy Bird AI")
            self.font = pygame.font.SysFont(None, 36)

        #Clock and reset to first state
        self.clock = pygame.time.Clock()
        self.reset()
    
#--------------------------------------------------------------------------------------------------------------------------

    #Resets the envrionment to start state
    def reset(self):
        #Reset bird and scores
        self.bird_y = self.HEIGHT // 2  #Bird position
        self.bird_velocity = 0  #Start bird with no speed
        self.frame_counter = 0  #Reset frame counter for score
        self.score = 0          #Reset score
        self.done = False       #Game is not over yet therefore don't close window

        #Reset pipes to offscreen, *similar to game.py*
        self.pipes = [] #Empty list of pipes
        for i in range(3):  #Creates 3 sets of pipes 
            with_item = (i == 1)
            self.pipes.append(self._create_pipe(self.WIDTH + i * self.pipe_spacing, with_item=with_item))

        return self._get_state()    #Return initial state (array)
    
#--------------------------------------------------------------------------------------------------------------------------

    #Creates a pipe, *similar to game.py*
    def _create_pipe(self, x_pos, with_item=False): #(x-cord of where pipe will appear, boolean if pipe has item)
        top = random.randint(50, self.HEIGHT - self.pipe_gap - 50)  #Random length of top pipe 
        bottom = top + self.pipe_gap    #Calculate length of bottom pipe
        item = None 
        if with_item:   #If there is an item
            item_x = x_pos + self.pipe_width // 2   #Places item inbetween pipe x-axis
            item_y = (top + bottom) // 2            #Places item inbetween pipe y-axis
            item = {'x': item_x, 'y': item_y, 'active': True}   #Creates item object with cords and collection status

        return {'x': x_pos, 'top': top, 'bottom': bottom, 'item': item, 'passed': False}
    
#--------------------------------------------------------------------------------------------------------------------------

    #Returns current state of game
    def _get_state(self):
        #Finds next pipe that is not passed -> checks the rightmost edge of each pipe, first one found is saved as closest pipe
        closest_pipe = next((p for p in self.pipes if p['x'] + self.pipe_width > self.bird_x), self.pipes[0])   

        item_dx = 0 #x distance to item
        item_dy = 0 #y distance to item
        #If closest pipe has item
        if closest_pipe['item'] and closest_pipe['item']['active']:
            item_dx = closest_pipe['item']['x'] - self.bird_x   #Find x distance to item
            item_dy = closest_pipe['item']['y'] - self.bird_y   #Find y distance to item

        #Returns info
        return np.array([
            self.bird_y,    #Bird y position
            self.bird_velocity, #Bird velocity
            closest_pipe['x'] - self.bird_x,    #How close horizontally to pipe
            closest_pipe['top'] - self.bird_y,   #How close vertically to pipe
            item_dx,    #Bird x distance to item
            item_dy     #Bird y distance to item
        ], dtype=np.float32)
    
#--------------------------------------------------------------------------------------------------------------------------

    #Function to dictate what happens after jump or not
    def step(self, action):
        #If jump
        if action == 1:
            self.bird_velocity = self.JUMP_STRENGTH #Negative velocity (jump)

        #Update bird's velocity with gravity, frame + 1
        self.bird_velocity += self.GRAVITY
        self.bird_y += self.bird_velocity
        self.frame_counter += 1

        #After 1 second (60 frames), add 1 to score
        if self.frame_counter >= self.FPS:
            self.score += 1
            self.frame_counter = 0

        #Move pipe and items left (similar to game.py)
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed
            if pipe['item'] and pipe['item']['active']:
                pipe['item']['x'] -= self.pipe_speed

        #Recycle old pipes, (similar to game.py)
        for i in range(len(self.pipes)):
            if self.pipes[i]['x'] + self.pipe_width < 0:
                farthest_x = max(p['x'] for p in self.pipes)
                with_item = random.random() < 0.5
                self.pipes[i] = self._create_pipe(farthest_x + self.pipe_spacing, with_item=with_item)

        #If pipe passed, +5 to score (similar to game.py)
        for pipe in self.pipes:
            if not pipe['passed'] and pipe['x'] + self.pipe_width < self.bird_x:
                self.score += 5
                pipe['passed'] = True

        #If bird overlaps item, +10 to score (similar to game.py)
        for pipe in self.pipes:
            if pipe['item'] and pipe['item']['active']:
                dx = self.bird_x - pipe['item']['x']
                dy = self.bird_y - pipe['item']['y']
                distance = (dx ** 2 + dy ** 2) ** 0.5
                if distance < self.bird_radius + self.item_radius:
                    self.score += 10
                    pipe['item']['active'] = False

        #If collision with top or bottom of screen, game over with -100 penalty
        if self.bird_y - self.bird_radius < 0 or self.bird_y + self.bird_radius > self.HEIGHT:
            self.done = True
            return self._get_state(), -100, True

        #If collision with pipes, game over with -100 penalty (similar to game.py)
        for pipe in self.pipes:
            if (
                pipe['x'] < self.bird_x + self.bird_radius < pipe['x'] + self.pipe_width and
                (self.bird_y - self.bird_radius < pipe['top'] or self.bird_y + self.bird_radius > pipe['bottom'])
            ):
                self.done = True
                return self._get_state(), -100, True

        return self._get_state(), 0.1, self.done
    
#--------------------------------------------------------------------------------------------------------------------------

    #Function to display current state (similar to game.py)
    def render(self):
        if not self.render_mode:
            return

        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)

        self.clock.tick(self.FPS)
        self.screen.fill(WHITE)

        pygame.draw.circle(self.screen, BLACK, (self.bird_x, int(self.bird_y)), self.bird_radius)

        for pipe in self.pipes:
            pygame.draw.rect(self.screen, GREEN, (pipe['x'], 0, self.pipe_width, pipe['top']))
            pygame.draw.rect(self.screen, GREEN, (pipe['x'], pipe['bottom'], self.pipe_width, self.HEIGHT - pipe['bottom']))
            if pipe['item'] and pipe['item']['active']:
                pygame.draw.circle(self.screen, RED, (int(pipe['item']['x']), int(pipe['item']['y'])), self.item_radius)

        score_surface = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_surface, (10, 10))

        pygame.display.flip()
