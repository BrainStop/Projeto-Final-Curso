import os
import gym
import neat
import pickle
from gym import wrappers
import numpy as np
import visualize

runs_per_net = 6

env = gym.make('MountainCar-v0')

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))
print(env.observation_space.high)
print(env.observation_space.low)

time_constant = 0.01

# env = gym.wrappers.Monitor(env, '/tmp/cartpole_v2_ff', force=True)

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = []
    state = env.reset()
    while True:
        output = net.activate(state)[0]
        if output >= 0.5:
            action = 0
        else:
            action = 2
        state, _, is_over, _ = env.step(action)
        fitness.append(state[0])

        if state[0] >= 0.6:
            return 10
            break

        if is_over:
            break

    return np.amax(fitness) - np.amin(fitness)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    # Load the file, witch is assumed to live in the same directory as this script
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'cfg.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    best_genome = pop.run(eval_genomes, 100)
    
    visualize.draw_net(config, best_genome, view=False,
                       filename=str(pop.generation) + "net")
    visualize.plot_stats(stats,
                         filename=str(pop.generation) + "plot_fit.png")
    visualize.plot_species(stats,
                           filename=str(pop.generation) + "plot_spec.png")

    with open('winner_ff.p', 'wb') as f:
        pickle.dump(best_genome, f)
        
def record_winner():
    with open('winner_ff.p', 'rb') as f:
        winner = pickle.load(f)
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'cfg.txt')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        net = neat.nn.FeedForwardNetwork.create(winner, config)
        monitor = wrappers.Monitor(env, '/tmp/cartpole_experiment', force=True)

        inputs = monitor.reset()
        i = 0
        while True:
            output = net.activate(inputs)[0]
            
            if output >= 0.5:
                action = 0
            else:
                action = 2
            i+= 1
            inputs, reward, is_over, _ = monitor.step(action)
            
            if is_over:
                break
        

if __name__ == '__main__':
    run()
#    record_winner()