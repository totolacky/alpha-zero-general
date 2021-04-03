import logging

import coloredlogs

from JanggiCoach import JanggiCoach as Coach
from janggi.JanggiGame import JanggiGame as Game
from janggi.pytorch.NNet import NNetWrapper as nn
from utils import *

import torch

import JanggiMainConstants as JMC

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': JMC.numIters,
    'numEps': JMC.numEps,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': JMC.tempThreshold,        #
    'updateThreshold': JMC.updateThreshold,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': JMC.maxlenOfQueue,    # Number of game examples to train the neural networks.
    'numMCTSSims': JMC.numMCTSSims,          # Number of games moves for MCTS to simulate.
    'arenaCompare': JMC.arenaCompare,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': JMC.cpuct,

    'checkpoint': JMC.checkpoint,
    'load_model': JMC.load_model,
    'load_folder_file': JMC.load_folder_file,
    'numItersForTrainExamplesHistory': JMC.numItersForTrainExamplesHistory,
    'checkpoint_folder': JMC.checkpoint_folder,

    'num_gpu_procs': JMC.num_gpu_procs,
    'num_selfplay_procs': JMC.num_selfplay_procs,
    'gpus_to_use': JMC.gpus_to_use,

    'remote_send': JMC.remote_send,
    'send_proc_params': JMC.send_proc_params,
    'recv_proc_params': JMC.recv_proc_params
})


def main():
    log.info('GPU availability: %s', torch.cuda.is_available())
    log.info('Loading %s...', Game.__name__)
    g = Game(0, 0)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

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
