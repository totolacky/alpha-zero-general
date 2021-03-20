import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

import torch.multiprocessing as mp
from torch.multiprocessing import Pool

from checkers.pytorch.NNet import NNetWrapper as nn

log = logging.getLogger(__name__)

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
# class NDPool(Pool):
#     Process = NoDaemonProcess

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        # self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        # self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    @staticmethod
    def executeEpisode(eeArgs):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        game, args, pipe_conn = eeArgs

        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        mcts = MCTS(game, pipe_conn, args)   # MCTS takes a pipe connection instead of nnet
        # mcts = MCTS(game, nnet, args)

        while True:
            episodeStep += 1
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)

            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, curPlayer, action)

            r = game.getGameEnded(board, curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
    
    @staticmethod
    def nnProcess(nnProcArgs):
        """
        
        """
        game, state_dict, selfplay_pipes, kill_pipe = nnProcArgs

        nnet = nn(game, state_dict)

        while True:
            # Check for incoming queries from all pipes
            for conn in selfplay_pipes:
                # If there are some, take care of them
                if (conn.poll()):
                    canonicalBoard = conn.recv()
                    s, v = nnet.predict(canonicalBoard)
                    conn.send((s, v))

            # Check for kill signal, through kill_pipe
            if (kill_pipe.poll()):
                assert kill_pipe.recv() == True
                return


    @staticmethod
    def parallelExecute(peArgs):
        """
        
        """
        game, args, state_dict, q = peArgs
        numEps = args.numEps
        num_selfplay_procs = args.num_selfplay_procs

        res = []

        for i in range(int(numEps / num_selfplay_procs)):
            # Create pipes
            nnC1, nnC2 = mp.Pipe()  # pipes for killing nnProcess
            pipes = []  # pipes for nnProcess
            pArgs = []  # pool args (with pipes for selfplayProcess)

            for j in range(num_selfplay_procs):
                c1, c2 = mp.Pipe()
                pipes.append(c1)
                pArgs.append((game, args, c2))

            # Create the NN process
            nnProc = mp.Process(target=Coach.nnProcess, args=[(game, state_dict, pipes, nnC2)])
            nnProc.daemon = True
            nnProc.start()

            # Create self-play processes (with pool)
            p = Pool(num_selfplay_procs)

            # Join and append the results to res
            for d in tqdm(p.map(Coach.executeEpisode, pArgs), total=len(pArgs)):
                res += d
            p.close()

            # Kill the NN process
            nnC1.send(True)
            nnProc.join()

        # Return res
        q.put(res)
        return

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                try:
                    mp.set_start_method('spawn')
                except RuntimeError:
                    pass

                procArg = []
                q = mp.Queue()
                state_dict = {k: v.cpu() for k, v in self.nnet.nnet.state_dict().items()}

                procs = []
                for j in range(self.args.num_gpu_procs):
                    p = mp.Process(target = self.parallelExecute, args=((self.game, self.args, state_dict, q),))
                    p.start()
                    procs.append(p)
                
                for p in procs:
                    p.join()

                for j in range(self.args.num_gpu_procs):
                    iterationTrainExamples += q.get()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            log.info('TRAINING AND SAVING NEW MODEL')
            self.nnet.train(trainExamples)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
