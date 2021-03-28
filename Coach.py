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
import torch
from time import time

from checkers.pytorch.NNet import NNetWrapper as nn

import socket

log = logging.getLogger(__name__)
fh = logging.FileHandler('temp/trainlog.txt')
log.addHandler(fh)

# class NoDaemonProcess(mp.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#     def _set_daemon(self, value):
#         pass
#     daemon = property(_get_daemon, _set_daemon)

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
        self.selfPlaysPlayed = 0

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
        # log.info('executeEpisode')
        # game, args, pipe_conn = eeArgs
        game, args, sharedQ = eeArgs

        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        pipeSend, pipeRecv = mp.Pipe()

        # mcts = MCTS(game, pipe_conn, args, True)   # MCTS takes a pipe connection instead of nnet
        mcts = MCTS(game, sharedQ, args, True, pipeSend, pipeRecv)   # MCTS takes a shared queue instead of nnet
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
                # log.info(str(len(trainExamples))+" examples generated")
                return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
    
    @staticmethod
    def nnProcess(nnProcArgs):
        """
        
        """
        # log.info('nnProcess')
        # game, state_dict, selfplay_pipes, kill_pipe = nnProcArgs
        game, state_dict, sharedQ, gpu_num = nnProcArgs

        nnet = nn(game, state_dict, gpu_num)

        while True:
            req = sharedQ.get()
            if req == None:
                return
            else:
                canonicalBoard, pipe = req
                s, v = nnet.predict(canonicalBoard)
                pipe.send((s, v))

            # # Check for incoming queries from all pipes
            # for conn in selfplay_pipes:
            #     # If there are some, take care of them
            #     if (conn.poll()):
            #         log.info("[nnProcess] Block 1")
            #         canonicalBoard = conn.recv()
            #         log.info("[nnProcess] Unblock 1")
            #         s, v = nnet.predict(canonicalBoard)
            #         conn.send((s, v))

            # # Check for kill signal, through kill_pipe
            # if (kill_pipe.poll()):
            #     log.info("[nnProcess] Block 2")
            #     assert kill_pipe.recv() == True
            #     log.info("[nnProcess] Unlock 2")
            #     return

    # @staticmethod
    # def parallelExecute(peArgs):
    #     """
        
    #     """
    #     game, args, state_dict, q = peArgs
    #     numEps = args.numEps
    #     num_selfplay_procs = args.num_selfplay_procs

    #     res = []

    #     # Create self-play processes (with pool)
    #     p = Pool(num_selfplay_procs)

    #     # Create pipes
    #     nnC1, nnC2 = mp.Pipe()  # pipes for killing nnProcess
    #     pipes = []  # pipes for nnProcess
    #     pArgs = []  # pool args (with pipes for selfplayProcess)

    #     for j in range(num_selfplay_procs):
    #         c1, c2 = mp.Pipe()
    #         pipes.append(c1)
    #         pArgs.append((game, args, c2))

    #     # Create the NN process
    #     nnProc = mp.Process(target=Coach.nnProcess, args=[(game, state_dict, pipes, nnC2)])
    #     nnProc.daemon = True
    #     nnProc.start()

    #     for i in range(int(numEps / num_selfplay_procs)):
    #         starttime = time()
            
    #         # Execute self-play and append the results to res
    #         for d in p.map(Coach.executeEpisode, pArgs):
    #             res += d
            
    #         endtime = time()
    #         print("Executed "+str(num_selfplay_procs)+" self-play episodes in "+str(endtime-starttime)+" secconds. (AVG "+str((endtime-starttime)/num_selfplay_procs)+" secs./game)")

    #     # Kill the NN process
    #     nnC1.send(True)
    #     nnProc.join()

    #     p.close()

    #     # Return res
    #     q.put(res)
    #     return

    @staticmethod
    def remoteSendProcess(rsProcArgs):
        """
        
        """
        HOST = 'eelabg13.kaist.ac.kr'
        PORT = 80
        result_conn = rsProcArgs

        # Create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server_socket.bind((HOST, PORT))
        server_socket.listen()

        # Return a new socket when a client connects
        client_socket, addr = server_socket.accept()

        # Address of connected client
        print('Connected by', addr)

        while True:
            # 클라이언트가 보낸 메시지를 수신하기 위해 대기합니다. 
            data = client_socket.recv(1024)

            # 빈 문자열을 수신하면 루프를 중지합니다. 
            if not data:
                break

            # 수신받은 문자열을 출력합니다.
            print('Received from', addr, data.decode())

            # 받은 문자열을 다시 클라이언트로 전송해줍니다.(에코) 
            client_socket.sendall(data)

        # 소켓을 닫습니다.
        client_socket.close()
        server_socket.close()


    @staticmethod
    def remoteRecvProcess(rrProcArgs):
        """
        
        """
        HOST = 'eelabg13.kaist.ac.kr'
        PORT = 80
        result_conn = rrProcArgs

        # Create a socket object
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        client_socket.connect((HOST, PORT))

        # Send a message
        client_socket.sendall('Hi'.encode())

        # Receive a message
        data = client_socket.recv(1024)
        print('Received', repr(data.decode()))

        # Close the socket
        client_socket.close()


    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        manager = mp.Manager()
        sharedQ = manager.Queue()

        # Create the server-communicating process
        remoteconn, remoteconn1 = mp.Pipe()
        rrProc = mp.Process(target=Coach.remoteSendProcess if self.args.remote_send else Coach.remoteRecvProcess, args=(remoteconn1, ))
        rrProc.daemon = True
        rrProc.start()

        # Generate self-plays and train
        for i in range(1, self.args.numIters + 1):
            # Create num_gpu_procs nnProcess
            nnProcs = []
            for j in range(self.args.num_gpu_procs):
                # Run nnProc
                state_dict = {k: v.cpu() for k, v in self.nnet.nnet.state_dict().items()}
                nnProc = mp.Process(target=Coach.nnProcess, args=[(self.game, state_dict, sharedQ, j%torch.cuda.device_count())])
                nnProc.daemon = True
                nnProc.start()
                nnProcs.append(nnProc)

            # Create self-play process pool
            # selfplayPool = Pool(self.args.num_selfplay_procs)
            selfplayPool = Pool(None)

            # Create pool args
            pArgs = []
            # for j in range(self.args.num_selfplay_procs):
            for j in range(self.args.numEps):
                # pArgs.append((self.game, self.args, spPipes[j]))
                pArgs.append((self.game, self.args, sharedQ))

            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                log.info('Start generating self-plays')

                with tqdm(total = self.args.numEps) as pbar:
                    for d in tqdm(selfplayPool.imap_unordered(Coach.executeEpisode, pArgs)):
                        iterationTrainExamples += d
                        pbar.update()
                
                self.selfPlaysPlayed += self.args.numEps

                # for t in tqdm(range(int(self.args.numEps / self.args.num_selfplay_procs))):
                #     for d in selfplayPool.map(Coach.executeEpisode, pArgs):
                #         iterationTrainExamples += d

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            # Kill the NN processes
            for j in range(self.args.num_gpu_procs):
                # nnPipes[j].send(True)
                sharedQ.put(None)

            for j in range(self.args.num_gpu_procs):
                nnProcs[j].join()
            
            # Close the process pool
            selfplayPool.close()

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
