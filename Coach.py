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

import socket, pickle

log = logging.getLogger(__name__)
fh = logging.FileHandler('temp/trainlog.txt')
log.addHandler(fh)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
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
        game, args, sharedQ = eeArgs

        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        mcts = MCTS(game, sharedQ, args, True)   # MCTS takes a shared queue instead of nnet

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

    @staticmethod
    def remoteSendProcess(rsProcArgs):
        """
        
        """
        HOST = 'eelabg13.kaist.ac.kr'
        PORT = 8080
        # result_conn = rsProcArgs
        dataQ, SDQ = rsProcArgs

        # Create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server_socket.bind((HOST, PORT))
        server_socket.listen()

        log.info('Socket started listening on port '+str(PORT))

        # Return a new socket when a client connects
        client_socket, addr = server_socket.accept()

        # Address of connected client
        log.info('Socket connected by'+str(addr))

        # Set socket timeout
        client_socket.settimeout(1.0)

        while True:
            # Receive a generated data
            # data = result_conn.recv()
            data = dataQ.get()

            # Send the data through socket
            client_socket.sendall(pickle.dumps(data))
            client_socket.sendall("This is the end of a pickled data.".encode())

            # Check if any state_dict arrived through the socket
            try:
                # sd = client_socket.recv(52428800)
                data = []
                while True:
                    packet = client_socket.recv(4096)
                    if packet[-34:]=="This is the end of a pickled data.".encode(): 
                        data.append(packet[:-34])
                        break
                    data.append(packet)
                state_dict = pickle.loads(b"".join(data))
                SDQ.put(state_dict)
                # result_conn.send(state_dict)

            except socket.timeout:
                continue
            
            # # If a state_dict arrived, pass it on to the main thread
            # state_dict = pickle.loads(sd)
            # result_conn.send(state_dict)

        # Close the socket
        client_socket.close()
        server_socket.close()


    @staticmethod
    def remoteRecvProcess(rrProcArgs):
        """
        
        """
        HOST = 'eelabg13.kaist.ac.kr'
        PORT = 8080
        # result_conn = rrProcArgs
        dataQ, SDQ = rrProcArgs

        # Create a socket object
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        client_socket.connect((HOST, PORT))
        log.info('Socket connected to host')

        while True:
            log.info('Waiting to receive data')
            # Receive a data
            data = []
            while True:
                packet = client_socket.recv(4096)
                # log.info('Received something: '+str(packet))
                # log.info('Msg: '+str(packet[-36:])+', is it the end? '+str())
                if packet[-34:]=="This is the end of a pickled data.".encode(): 
                    data.append(packet[:-34])
                    break
                data.append(packet)
            # data = client_socket.recv(52428800)
            data = pickle.loads(b"".join(data))

            log.info('Received data of length '+str(len(data)))

            # Send the data over the pipe
            # result_conn.send(data)
            dataQ.put(data)

            # Check if any state_dict arrived, and send it over the socket
            log.info('Empty: '+str(SDQ.empty()))
            if not SDQ.empty():
                sd = SDQ.get()
                while not SDQ.empty():
                    sd = SDQ.get()
            # if result_conn.poll():
            #     sd = result_conn.recv()
            #     # If more than 2 state_dicts are waiting, then send the most recent one
            #     while result_conn.poll():
            #         sd = result_conn.recv()
                client_socket.send(pickle.dumps(sd))
                client_socket.send("This is the end of a pickled data.".encode())

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
        # remoteconn, remoteconn1 = mp.Pipe()
        remoteDataQ = manager.Queue()
        remoteSDQ = manager.Queue()
        # rrProc = mp.Process(target=Coach.remoteSendProcess if self.args.remote_send else Coach.remoteRecvProcess, args=(remoteconn1, ))
        rrProc = mp.Process(target=Coach.remoteSendProcess if self.args.remote_send else Coach.remoteRecvProcess, args=((remoteDataQ, remoteSDQ),))
        rrProc.daemon = True
        rrProc.start()

        # Generate self-plays and train
        for i in range(1, self.args.numIters + 1):
            # If remote_send (i.e. Haedong server), update state_dict
            if self.args.remote_send:
                log.info("Checking for state_dict update")
                if not remoteQ.empty():
                    sd = remoteQ.get()
                    while not remoteQ.empty():
                        sd = remoteQ.get()
                # if remoteconn.poll():
                #     sd = remoteconn.recv()
                #     while remoteconn.poll():
                #         sd = remoteconn.recv()
                    self.nnet.nnet.load_state_dict(sd)
                    log.info("Updated state_dict")
                else:
                    log.info("No new state_dict available")

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
            selfplayPool = Pool(None)

            # Create pool args
            pArgs = []
            for j in range(self.args.numEps):
                pArgs.append((self.game, self.args, sharedQ))

            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                log.info('Start generating self-plays')

                with tqdm(total = self.args.numEps) as pbar:
                    for d in tqdm(selfplayPool.imap_unordered(Coach.executeEpisode, pArgs)):
                        if self.args.remote_send:
                            remoteDataQ.put(d)
                            # remoteconn.send(d)
                        else:
                            iterationTrainExamples += d
                        pbar.update()
                
                self.selfPlaysPlayed += self.args.numEps

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            # Close the process pool
            selfplayPool.close()

            # Kill the NN processes
            for j in range(self.args.num_gpu_procs):
                sharedQ.put(None)

            for j in range(self.args.num_gpu_procs):
                nnProcs[j].join()

            # If the process is remote_send (i.e. the Haedong server), then skip the training part
            if self.args.remote_send:
                continue
            
            # Otherwise, add the server-generated examples to the iterationTrainExamples
            num_remote_selfplays = 0
            while not remoteDataQ.empty():
                d = remoteDataQ.get()
            # while remoteconn.poll():
            #     d = remoteconn.recv()
                iterationTrainExamples += d
                num_remote_selfplays += 1
            
            log.info(f'{num_remote_selfplays} self-play data loaded from remote server')
            self.selfPlaysPlayed += num_remote_selfplays

            # Update the trainExamplesHistory
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
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(self.selfPlaysPlayed))

            # Send the new state_dict
            state_dict = {k: v.cpu() for k, v in self.nnet.nnet.state_dict().items()}
            # remoteconn.send(state_dict)
            remoteSDQ.put(state_dict)

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
