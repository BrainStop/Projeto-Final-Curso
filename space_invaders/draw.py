#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 11:04:54 2017

@author: brain_stop
"""

import neat
import visualize

def run():
    # Load the file, witch is assumed to live in the same directory as this script
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'conf_ff.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    winner = pop.run(eval_genomes, 300)
    node_names = {-1:'cPos', -2: 'cVel', -3: 'pAng', -4: 'pAngVel', 0:'esq ou dir'}
    visualize.draw_net(config, winner, False, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    with open('winner_ff.p', 'wb') as f:
        pickle.dump(winner, f)
