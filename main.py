import logging

import coloredlogs

from Coach import Coach
from checkers.CheckersGame import CheckersGame as Game
from checkers.pytorch.NNet import NNetWrapper as nn
from utils import *

import torch

log = logging.getLogger(__name__)
fh = logging.FileHandler('temp/trainlog.txt')
log.addHandler(fh)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 10000,
    'numEps': 32,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 80,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 0,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','checkpoint_60.pth.tar'),
    'numItersForTrainExamplesHistory': 100,

    'num_gpu_procs': 3,
    'remote_send': False
})


def main():
    log.info('GPU availability: %s', torch.cuda.is_available())
    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, None)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()