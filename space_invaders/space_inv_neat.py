#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:29:46 2017

@author: brain_stop
"""

import os
import pickle
import multiprocessing
import gym
from gym import wrappers
import neat
import visualize
import numpy as np


NUM_CORES = 8
RENDER = True
GAME = "SpaceInvaders-ram-v0"
RUNS_PER_NET = 3
MAX_GEN = 5
SOLVED_SCORE = 36
# SIGMA_SCALING = False
# BOLTZMAN_SCALING = False
FILTER = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
          60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 6, 16, 17, 18, 19, 20, 21,
          22, 23, 26, 81, 82, 83, 84, 85, 87]

# ===============
# Neat Functions
# ===============


def run():
    """run"""
    # Load the file, witch is assumed
    # to live in the same directory as this script
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'cfg.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pool_evl_gen = PooledEvalGenome(NUM_CORES, GAME, RUNS_PER_NET)
    env = gym.make(GAME)

    end = False
    while True:
        best_genome = pop.run(pool_evl_gen.eval_genomes, MAX_GEN)
        visualize.draw_net(config, best_genome, view=False,
                           filename=str(pop.generation) + "net")
        visualize.plot_stats(stats,
                             filename=str(pop.generation) + "plot_fit.png")
        visualize.plot_species(stats,
                               filename=str(pop.generation) + "plot_spec.png")
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        fitness = simulation(env, net, RUNS_PER_NET, RENDER)
        if fitness >= SOLVED_SCORE:
            end = True
            break

        with open(str(pop.generation) + 'pop.p', 'wb') as file:
            pickle.dump((pop, stats), file)

        if end:
            break


def simulation(env, net, runs_per_net=1, render=False):
    """simulation"""
    fitnesses = []
    for _ in range(runs_per_net):
        fitness = 0.0
        state = env.reset()
        prev_state = 36
        while True:
            action = 0
            filtered_state = state[FILTER]
            output = net.activate(filtered_state)
            action = ouput_to_action(output)
            state, reward, is_over, _ = env.step(action)
            if state[17] != 0:
                if prev_state != state[17]:
                    prev_state = state[17]
                    fitness = fitness + 1
            if render:
                env.render()
            if is_over:
                break

        fitnesses.append(fitness)
    return min(fitnesses)


def replay(filename, game=GAME, record=False):
    """"replay"""
    with open(filename, 'rb') as file:
        pop = pickle.load(file)[0]
        genome = pop.best_genome
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'cfg.txt')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

        env = gym.make(game)
        print("genome fitness: ", genome.fitness)
        if record:
            env = wrappers.Monitor(env, '/tmp/'+game)
#        net = neat.nn.FeedForwardNetwork.create(genome, pop.config)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = simulation(env, net, render=True)
        print("simulation fitness: ", fitness)


class PooledEvalGenome(object):
    """Runs the evaluation function in parallel acording to the number of
    cores"""
    def __init__(self, num_cores, game, runs_per_net, render=False):
        self.game = game
        self.render = render
        self.num_cores = num_cores
        self.runs_per_net = runs_per_net
        self.enviroments = [gym.make(self.game) for _ in range(self.num_cores)]

    def eval_genomes(self, genomes, config):
        """Evaluation function"""
        pool = multiprocessing.Pool(self.num_cores)
        jobs = []
        for gen_chunk, env in zip(np.array_split(genomes, self.num_cores),
                                  self.enviroments):
            jobs.append(pool.apply_async(self.eval_thread,
                                         (gen_chunk, config, env)))

        total_fitnesses = []
        for job in jobs:
            total_fitnesses = np.hstack((total_fitnesses,
                                         job.get(timeout=None)))
        pool.close()

        for genome, fitness in zip(genomes, total_fitnesses):
            genome[1].fitness = fitness

    def eval_thread(self, genomes, config, env):
        """Function used in each thread"""
        return [self.eval_genome(genome, config, env) for _, genome in genomes]

    def eval_genome(self, genome, config, env):
        """Evaluates the genome anda returns his fitness value"""
        # Instancia um objeto do tipo feedforward neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = simulation(env, net, self.runs_per_net, self.render)
        return fitness


def ouput_to_action(output):
    np_output = np.array(output)
    activation_tresh = 0.9
    output_thresh = list((np_output > activation_tresh) * 1)
    if output_thresh == [0, 0, 0]:
        return 0
    if output_thresh == [0, 1, 0]:
        return 1
    if output_thresh == [1, 0, 0]:
        return 3
    if output_thresh == [0, 0, 1]:
        return 2
    if output_thresh == [1, 1, 0]:
        return 5
    if output_thresh == [0, 1, 1]:
        return 4
    if output_thresh == [1, 0, 1] or output_thresh == [1, 1, 1]:
        if output[0] >= output[2]:
            if output[1] >= activation_tresh:
                return 5
            else:
                return 3
        else:
            if output[1] >= activation_tresh:
                return 4
            else:
                return 2

run()