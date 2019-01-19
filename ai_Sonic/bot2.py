import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
imgarray = []
xpos_end = 0

def eval_genomes(genomes, config):
 for genome_id, genome in genomes:
  observ = env.reset()
  action = env.action_space.sample()

  inx, iny, inc = env.observation_space.shape  #the x, y, and colors
  inx = int(inx/8)
  iny = int(iny/8)

  net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

  current_max_fitness = 0
  fitness_current =  0
  frame = 0
  counter = 0
  xpos = 0
  xpos_max = 0

  done = False

  while not done:
      env.render()
      frame += 1

      observ = cv2.resize(observ, (inx, iny))
      observ = cv2.cvtColor(observ, cv2.COLOR_BGR2GRAY)
      observ = np.reshape(observ, (inx,iny)) 

      imgarray = np.ndarray.flatten(observ)
      nnOutput = net.activate(imgarray)
      observ, reward, done, info = env.step(nnOutput)
      
      xpos = info['x']
      xpos_end = info['screen_x_end']
      
      if xpos > xpos_max:
          fitness_current += 1
          xpos_max = xpos
      
      if xpos == xpos_end and xpos > 500:
          fitness_current += 100000
          done = True
      
      fitness_current += reward
      
      if fitness_current > current_max_fitness:
          current_max_fitness = fitness_current
          counter = 0
      else:
          counter += 1
          
      if done or counter == 250:
          done = True
          print(genome_id, fitness_current)
          
      genome.fitness = fitness_current

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

population = neat.Population(config)

population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(10))

winner = population.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)