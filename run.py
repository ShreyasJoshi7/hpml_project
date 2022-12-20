
import pygame
import random
import os
import time
import neat
import graphics_and_plots
import pickle
pygame.font.init()

WIDTH = 700
HEIGHT = 900
FLOOR_GAME = 730
START_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
LINES = False

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

generation = 0

class Base:
    Velocity = 5
    Width = base_img.get_width()
    image = base_img
    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.Width
    def move(self):
        self.x1 -= self.Velocity
        self.x2 -= self.Velocity
        if self.x1 + self.Width < 0:
            self.x1 = self.x2 + self.Width
        if self.x2 + self.Width < 0:
            self.x2 = self.x1 + self.Width
    def draw(self, win):
        win.blit(self.image, (self.x1, self.y))
        win.blit(self.image, (self.x2, self.y))

class Pipe():
    Gap = 200
    Velocity = 5
    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        self.passed = False
        self.set_height()
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.Gap
    def move(self):
        self.x -= self.Velocity
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))
    def collide(self, bird, win):
        mask_bird = bird.get_mask()
        masktop = pygame.mask.from_surface(self.PIPE_TOP)
        maskbott = pygame.mask.from_surface(self.PIPE_BOTTOM)
        offsettop = (self.x - bird.x, self.top - round(bird.y))
        offsetbott = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = mask_bird.overlap(maskbott, offsetbott)
        t_point = mask_bird.overlap(masktop,offsettop)
        if b_point or t_point:
            return True
        return False
    

class Bird:
    Maximum_rot = 25
    Images = bird_images
    Rotation_vel = 20
    time_animation = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.Images[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        displacement_bird = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2
        if displacement_bird >= 16:
            displacement_bird = (displacement_bird/abs(displacement_bird)) * 16
        if displacement_bird < 0:
            displacement_bird -= 2
        self.y = self.y + displacement_bird
        if displacement_bird < 0 or self.y < self.height + 50:
            if self.tilt < self.Maximum_rot:
                self.tilt = self.Maximum_rot
        else:
            if self.tilt > -90:
                self.tilt -= self.Rotation_vel

    def draw(self, win):
        self.img_count += 1
        if self.img_count <= self.time_animation:
            self.img = self.Images[0]
        elif self.img_count <= self.time_animation*2:
            self.img = self.Images[1]
        elif self.img_count <= self.time_animation*3:
            self.img = self.Images[2]
        elif self.img_count <= self.time_animation*4:
            self.img = self.Images[1]
        elif self.img_count == self.time_animation*4 + 1:
            self.img = self.Images[0]
            self.img_count = 0
        if self.tilt <= -80:
            self.img = self.Images[1]
            self.img_count = self.time_animation*2
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)
    def get_mask(self):
        return pygame.mask.from_surface(self.img)



def eval(genomes, config):
    global WINDOW, generation
    win = WINDOW
    generation += 1
    nets = []
    birds = []
    g = []
    for genome_id, genome in genomes:
        genome.fitness = 0 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.append(genome)

    base = Base(FLOOR_GAME)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    play = True
    while play and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  
                pipe_ind = 1                                                                

        for x, bird in enumerate(birds):  
            g[x].fitness += 0.1
            bird.move()
            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5: 
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for bird in birds:
                if pipe.collide(bird, win):
                    g[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    g.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            for genome in g:
                genome.fitness += 5
            pipes.append(Pipe(WIDTH))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR_GAME or bird.y < -50:
                nets.pop(birds.index(bird))
                g.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw(WINDOW, birds, pipes, base, score, generation, pipe_ind)




def draw(win, birds, pipes, base, score, generation, pipe_ind):
    if generation == 0:
        generation = 1
    win.blit(bg_img, (0,0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)
    for bird in birds:
        if LINES:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        bird.draw(win)
    score_label = START_FONT.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (WIDTH - score_label.get_width() - 15, 10))
    score_label = START_FONT.render("Gens: " + str(generation-1),1,(255,255,255))
    win.blit(score_label, (10, 10))
    score_label = START_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(score_label, (10, 50))
    pygame.display.update()

def train(feedfordward_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         feedfordward_file)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    vals = neat.StatisticsReporter()
    population.add_reporter(vals)
    winner = population.run(eval, 5)

    print('\nBest genome:\n{!s}'.format(winner))

def blitRotateCenter(surf, image, topleft, angle):
    image_rotated = pygame.transform.rotate(image, angle)
    rect_new = image_rotated.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(image_rotated, rect_new.topleft)
    
if __name__ == '__main__':
    path = os.path.dirname(__file__)
    feedfordward_file = os.path.join(path, 'config-feedforward.txt')
    train(feedfordward_file)







