import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
imgarray = []

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
        	observ = cv2.cvtColor(observ, cv2.COLOR_BGR2GrAY)
        	observ = np.reshape(observ, (inx, iny))

        	for x in observ:
        		for y in x:
        			imgarray.append(y)

        	nnOutput = net.activate(imgarray)

        	print(nnOutput)


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
	                 neat.DefaultSpeciesSet, neat.DefaultStagnation, 
	                 'config-feedforward')

population = neat.Population(config)

winner = population.run(eval_genomes)