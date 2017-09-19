import os
import gym
import neat
import pickle
from gym import wrappers
import visualize

runs_per_net = 6
env = gym.make('CartPole-v1')
time_constant = 0.01

# env = gym.wrappers.Monitor(env, '/tmp/cartpole_v2_ff', force=True)


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []
    for runs in range(runs_per_net):
        fitness = 0.0
        inputs = env.reset()
        while True:
            inputs = [inputs[0], inputs[1]]
            output = net.activate(inputs)[0]
            action = output > 0.5
            inputs, reward, is_over, _ = env.step(action)
            fitness += reward
            if is_over:
                break
        fitnesses.append(fitness)
    return min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


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



def record_winner():
    with open('winner_ff.p', 'rb') as f:
        winner = pickle.load(f)
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'conf_ff.txt')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        net = neat.nn.FeedForwardNetwork.create(winner, config)
        monitor = wrappers.Monitor(env, '/tmp/cartpole_experiment', force=True)

        inputs = monitor.reset()

        while True:
            inputs = [inputs[1], inputs[3]]
            output = net.activate(inputs)[0]
            action = output > 0.5
            inputs, reward, is_over, _ = monitor.step(action)

            if is_over:
                break

if __name__ == '__main__':
    run()
    