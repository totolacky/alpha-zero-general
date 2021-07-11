import logging
import os
import torch
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import struct

import numpy as np
from tqdm import tqdm

from JanggiArena import JanggiArena
from JanggiMCTS import JanggiMCTS

import torch.multiprocessing as mp	
from torch.multiprocessing import Pool	
from time import time, sleep
from janggi.pytorch.NNet import NNetWrapper as nn	
from janggi.JanggiGame import JanggiGame as Game	
import requests, pickle

import JanggiMainConstants as JMC

from janggi.JanggiConstants import *
from janggi.JanggiLogic import Board
from janggi.JanggiPlayers import *

log = logging.getLogger(__name__)

class JanggiCoach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, selfPlaysPlayed = 0):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.selfPlaysPlayed = selfPlaysPlayed

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
            trainExamples: a list of examples of the form (encodedBoard, pi, v) #(canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        game, args, sharedQ, mctsQ, mctsQIdx, nextSelfplayQ, state_dict = eeArgs

        trainExamples = []
        board = game.getInitBoard()
        episodeStep = 0
        alternate = 1

        # actionList = []

        if sharedQ == None:
            # nnet = nn(game, state_dict, mctsQIdx)
            mcts = JanggiMCTS(game, state_dict, args)
        else:
            mcts = JanggiMCTS(game, sharedQ, args, True, mctsQ, mctsQIdx)

        while True:
            episodeStep += 1
            encodedBoard = game.encodeBoard(board)
            temp = int(episodeStep < args.tempThreshold)

            pi = mcts.getActionProb(board, temp=temp)
            trainExamples.append([encodedBoard, alternate, pi, None])

            alternate = -alternate

            action = np.random.choice(len(pi), p=pi)
            board = game.getNextState(board, action)
            r = game.getGameEnded(board)

            if r != 0:
                data = [(x[0], x[2], r * x[1]) for x in trainExamples]
                if nextSelfplayQ != None:
                    nextSelfplayQ.put((True, (data, mctsQIdx)))
                return data

    @staticmethod
    def playGame(pgArgs):
        """
        Executes one episode of a game.

        Returns:
            winner: player who won the game (1 if player1, -1 if player2)
        """
        game, args, is_rp, is_p1, checkpoint, nextSelfPlayQ = pgArgs

        mcts = JanggiMCTS(g, state_dict, args)

        player1 = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
        player2 = RandomPlayer(game).play if is_rp else GreedyJanggiPlayer(game).play

        if not is_p1:
            tmp = player1
            player1 = player2
            player2 = tmp

        players = [player2, None, player1]
        curPlayer = 1
        board = game.getInitBoard()
        it = 0

        while game.getGameEnded(board) == 0:
            it += 1
            action = players[curPlayer + 1](board)

            valids = game.getValidMoves(board)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid! Current player is {curPlayer}')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board = game.getNextState(board, action)

            curPlayer *= -1

        nextSelfPlayQ.put((False, (checkpoint, is_rp, game.getGameEnded(board) * (1 if is_p1 else -1))))

        return 0

    @staticmethod
    def checkpointSCP(from_path, to_path):	
        """	
        	
        """	
        log.info('ACQUIRING LOCK FOR SCP')
        can_access = pickle.loads(requests.get(url = JMC.request_base_url+"/acquireLock").content)
        while (not can_access):
            sleep(10)
            can_access = pickle.loads(requests.get(url = JMC.request_base_url+"/acquireLock").content)

        os.system("scp "+ from_path + " " + to_path)

        requests.get(url = JMC.request_base_url+"/releaseLock")

    @staticmethod	
    def nnProcess(nnProcArgs):	
        """	
        	
        """	
        game, updatePipe, sharedQ, gpu_num, queues, checkpoint_folder = nnProcArgs
        should_update = False
        lastTime = 0

        nnet = nn(game, None, gpu_num)
        while True:	
            # Check for nn updates
            if updatePipe.poll():
                log.info("new checkpoint exists!")
                cp_name = updatePipe.recv()
                while updatePipe.poll():
                    cp_name = updatePipe.recv()
                log.info("cp_name: "+str(cp_name))
                should_update = True
                lastTime = time()

            # Update NN if possible
            if (should_update):
                if (time() - lastTime > 1):
                    lastTime = time()
                    log.info('ACQUIRING LOCK FOR MOUNTED FOLDER ACCESS')
                    can_access = pickle.loads(requests.get(url = JMC.request_base_url+"/acquireLock").content)
                    if (can_access):
                        should_update = False
                        with open(JanggiCoach.getSharedStateDictFile(JMC.checkpoint_folder), 'rb') as handle:
                            state_dict = pickle.load(handle)
                        nnet.nnet.load_state_dict(state_dict)
                        log.info('Updated network.')
                        updatePipe.send(0)
                        requests.get(url = JMC.request_base_url+"/releaseLock")
                    else:
                        log.info('FAILED TO ACQUIRE ACCESS')

            # Check for evaluation requests
            req = sharedQ.get()	
            if req == None:	
                return	
            else:
                # canonicalBoard, pipe = req	
                canonicalBoard, qIdx = req	
                s, v = nnet.predict(canonicalBoard)
                queues[qIdx].put((s,v))

    @staticmethod	
    def trainingHTTPProcess(rrProcArgs):
        """	
        	
        """	
        dataQ, base_url = rrProcArgs

        while True:
            # Receive a data point
            cnt, data = pickle.loads(requests.get(url = base_url+"/getData").content)
            dataQ.put((cnt, data))
            sleep(10)
    
    def learn(self):
        """
        Performs iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        try:
            mp.set_start_method('spawn')	
        except RuntimeError:	
            pass
        if not self.args.is_training_client:
            self.learn_selfplay_client()
        else:
            self.learn_training_only_client()

    def learn_selfplay_client(self):
        """
        Process that continuously generates self-play data
        """
        manager = mp.Manager()
        sharedQ = manager.Queue()
        statedict_name = "Default"

        # Create num_selfplay_procs queues for sending nn eval results to selfplay procs.
        queues = []
        for j in range(self.args.num_selfplay_procs):
            queues.append(manager.Queue())

        # Create num_gpu_procs queues for sending state_dict update info to nn procs.
        nn_update_pipes1 = []
        nn_update_pipes2 = []
        for j in range(self.args.num_gpu_procs):
            c1, c2 = mp.Pipe()
            nn_update_pipes1.append(c1)
            nn_update_pipes2.append(c2)

        # Create num_gpu_procs nnProcess
        nnProcs = []
        for j in range(self.args.num_gpu_procs):
            # Run nnProc
            nnProc = mp.Process(target=JanggiCoach.nnProcess, args=[(self.game, nn_update_pipes1[j], sharedQ, self.args.gpus_to_use[j%len(self.args.gpus_to_use)], queues, self.args.checkpoint_folder)])
            nnProc.daemon = True
            nnProc.start()
            nnProcs.append(nnProc)

        # Create a queue for receiving info of finished jobs
        nextSelfplayQ = manager.Queue()

        # Create self-play process pool
        selfplayPool = Pool(self.args.num_selfplay_procs)

        # Run the first num_selfplay_procs process
        ibs = pickle.loads(requests.get(url = self.args.request_base_url+"/getIBS").content)
        for j in range(self.args.num_selfplay_procs):
            selfplayPool.apply_async(JanggiCoach.executeEpisode, [(Game(self.game.c1, self.game.c2, mode = ibs), self.args, sharedQ, queues[j], j, nextSelfplayQ, None)])
        
        # Continuously generate self-plays
        while True:
            # Check for any network updates
            new_sd = pickle.loads(requests.get(url = self.args.request_base_url+"/getSD").content)
            if statedict_name != new_sd:
                statedict_name = new_sd

                sharedStateDictFile = JanggiCoach.getSharedStateDictFile(self.args.remote_checkpoint_folder)
                if (self.args.scp_base_url != None):
                    JanggiCoach.checkpointSCP(self.args.scp_base_url + ":" + sharedStateDictFile, sharedStateDictFile)

                for q in nn_update_pipes2:
                    q.send(statedict_name)
                    q.recv()
                log.info('Alerted the nn procs to update the network')

            # Wait for a selfplay result
            is_selfplay, q_data = nextSelfplayQ.get()
            if is_selfplay:
                data, finished_id = q_data
                self.selfPlaysPlayed += 1
                log.info(str(self.selfPlaysPlayed)+' selfplay games played. Data length = '+str(len(data)))
                requests.post(url = self.args.request_base_url+"/postData", data = pickle.dumps(data))
            else:
                checkpoint, is_rp, did_win = q_data
                log.info("Evaluated ("+str(checkpoint)+", "+str(is_rp)+", "+str(did_win)+")")
                requests.post(url = self.args.request_base_url+"/uploadEvalRes", data = pickle.dumps((checkpoint, is_rp, did_win)))

            # Run new selfplay
            ibs = pickle.loads(requests.get(url = self.args.request_base_url+"/getIBS").content)
            next_game = pickle.loads(requests.get(url = self.args.request_base_url+"/getNextGame").content)
            if next_game == None:
                selfplayPool.apply_async(JanggiCoach.executeEpisode, [(Game(self.game.c1, self.game.c2, mode = ibs), self.args, sharedQ, queues[finished_id], finished_id, nextSelfplayQ, None)])
            else:
                checkpoint, is_rp, is_p1 = next_game
                assert False


    def learn_training_only_client(self):
        """
        Process that only trains the network
        """
        untrained_cnt = 0
        i = 0

        # Load self-plays and train
        while True:
            i += 1
            log.info(f'Starting Iter #{i} ... ({self.selfPlaysPlayed} games played)')

            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            # Train a lot on the first trial to prevent infinite move masking
            if (i == 1):
                trainFreq = 0 if self.skipFirstSelfPlay else 100
            else:
                trainFreq = self.args.trainFrequency

            while (untrained_cnt < trainFreq):
                # Load self-plays from server
                c, d = pickle.loads(requests.get(url = self.args.request_base_url+"/getData").content)
                log.info(f'{c} self-play data loaded')
                self.selfPlaysPlayed += c
                untrained_cnt += c
                iterationTrainExamples += d
                if (untrained_cnt < trainFreq):
                    sleep(60)
            
            log.info(f'{untrained_cnt} GAMES LOADED: TRAINING AND SAVING NEW MODEL')

            # Add the server-generated examples to the iterationTrainExamples
            self.trainExamplesHistory.append(iterationTrainExamples)
            untrained_cnt = 0

            # Update the trainExamplesHistory
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            # Use at most maxDataCount data points for training
            data_cnt = 0
            for e in self.trainExamplesHistory:
                data_cnt += len(e)
            while data_cnt > self.args.maxDataCount:
                data_cnt -= len(self.trainExamplesHistory[0])
                self.trainExamplesHistory.pop(0)

            # backup history to a file every 10 iterations
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            if (i % 10 == 0):
                self.saveTrainExamples(self.selfPlaysPlayed)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            log.info('TRAINING AND SAVING NEW MODEL')	
            self.nnet.train(trainExamples)
            # Save checkpoints every iteration
            self.nnet.save_checkpoint(folder=self.args.checkpoint_folder, filename=self.getCheckpointFile(self.selfPlaysPlayed))
            
            log.info('ACQUIRING LOCK FOR MOUNTED FOLDER ACCESS')
            can_access = pickle.loads(requests.get(url = self.args.request_base_url+"/acquireLock").content)
            while (not can_access):
                sleep(10)
                can_access = pickle.loads(requests.get(url = self.args.request_base_url+"/acquireLock").content)

            log.info('SAVING CHECKPOINT')
            with open(JanggiCoach.getSharedStateDictFile(self.args.checkpoint_folder), 'wb') as handle:
                pickle.dump({k: v.cpu() for k, v in self.nnet.nnet.state_dict().items()}, handle)

            requests.get(url = self.args.request_base_url+"/releaseLock")

            # Send evaluation request
            requests.post(url = self.args.request_base_url+"/pushEval", data = pickle.dumps((False, self.selfPlaysPlayed)))

            # Send the new state_dict
            requests.post(url = self.args.request_base_url+"/updateSD", data = pickle.dumps(self.selfPlaysPlayed))
            log.info('Alerted updated network')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pickle'

    @staticmethod
    def getStateDictFile(self, folder, iteration):
        return os.path.join(folder, 'sd_' + str(iteration) + '.pickle')

    @staticmethod
    def getSharedStateDictFile(folder):
        return os.path.join(folder, 'sd_shared.pickle')

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint_folder
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
