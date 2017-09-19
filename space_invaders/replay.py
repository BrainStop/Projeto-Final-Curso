import gym
import neat
import pickle
import numpy as np
from gym import wrappers


def simulation(env, net, render=False):
    """simulation"""
    fitness = 0.0
    state = env.reset()
    while True:
        output = net.activate(state)
        action = np.argmax(output)
        state, reward, is_over, _ = env.step(action)
        fitness += reward
        if render:
            env.render()
        if is_over:
            break
    return fitness


def replay(filename, game="SpaceInvaders-ram-v0", seed_val=None, record=False):
    """"replay"""
    with open(filename, 'rb') as file:
        pop = pickle.load(file)
        genome = pop.best_genome
        env = gym.make(game)
        print("genome fitness: " + genome.fitness)
        if record:
            env = wrappers.Monitor(env, '/tmp/'+game)
        env.seed(seed=seed_val)
        net = neat.nn.FeedForwardNetwork.create(genome, pop.config)
        fitness = simulation(env, net, render=False)
        print("simulation fitness: " + fitness)
