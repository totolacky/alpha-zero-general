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
from time import time
from janggi.pytorch.NNet import NNetWrapper as nn	
from janggi.JanggiGame import JanggiGame as Game	
import socket, pickle

log = logging.getLogger(__name__)

class JanggiCoach():
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
            trainExamples: a list of examples of the form (encodedBoard, pi, v) #(canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        game, args, sharedQ, mctsQ, mctsQIdx, nextSelfplayQ = eeArgs

        trainExamples = []
        board = game.getInitBoard()
        episodeStep = 0
        alternate = 1

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
                    nextSelfplayQ.put((data, mctsQIdx))
                return data

    @staticmethod	
    def nnProcess(nnProcArgs):	
        """	
        	
        """	
        game, updatePipe_stateDict, sharedQ, gpu_num, queues, checkpoint_folder = nnProcArgs
        if checkpoint_folder == None:
            nnet = nn(game, updatePipe_stateDict, gpu_num)
        else:
            nnet = nn(game, None, gpu_num)
        while True:	
            # Check for nn updates
            if checkpoint_folder != None and updatePipe_stateDict.poll():
                cp_name = updatePipe_stateDict.recv()
                while updatePipe_stateDict.poll():
                    cp_name = updatePipe_stateDict.recv()
                with open(cp_name, 'rb') as handle:
                    state_dict = pickle.load(handle)
                nnet.nnet.load_state_dict(state_dict)
                log.info('Updated network.')
                updatePipe_stateDict.send(0)
            
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
    def recvSocket(channel, alert_timeout = False):
        """
        Receive exact bytes from socket
        """
        try:
            size_recv = channel.recv(struct.calcsize("i"))
            size = struct.unpack("i", size_recv)[0]
            data = b""
            size *= 3
            l = 0
            while len(data) < size:
                msg = channel.recv(min(4096, size - len(data)))
                if not msg:
                    return None
                data += msg
                l += len(msg)
            data = pickle.loads(data)
            return data
        except socket.timeout:
            if alert_timeout:
                print ("socket timeout while receiving")
            return None

    @staticmethod	
    def sendSocket(channel, msg, alert_timeout = True):
        """
        Send data through socket
        """
        msg = pickle.dumps(msg, pickle.HIGHEST_PROTOCOL)
        msgs = [msg[i:i+50000] for i in range(0, len(msg), 50000)]
        try:
            channel.send(struct.pack("i", len(msg)))
            for m in msgs:
                channel.send(m)
            return True
        except socket.timeout:
            if alert_timeout:
                print ("socket timeout while sending")
            return False

    @staticmethod	
    def remoteSendProcess(rsProcArgs):	
        """	
        	
        """	
        dataQ, SDQ, HOST, PORT, is_haedong, urp_available = rsProcArgs

        if urp_available:
            assert is_haedong

        # If haedong computer, create a server socket and two client sockets
        if is_haedong:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((HOST, PORT))	
            server_socket.listen()
            log.info('Socket started listening on port '+str(PORT))	

            # Return a new socket when each client connects	
            client_socket1, addr = server_socket.accept()
            log.info('Socket connected by'+str(addr))

            if urp_available:
                client_socket2, addr1 = server_socket.accept()
                log.info('Socket connected by'+str(addr1))

                msg1 = JanggiCoach.recvSocket(client_socket1)
                msg2 = JanggiCoach.recvSocket(client_socket2)

                if msg1 == 'RecvProc':
                    assert msg2 == 'SendProc'
                    client_socket = client_socket1  # The training machine
                    gen_socket = client_socket2     # The other generating machine
                else:
                    assert msg2 == 'RecvProc'
                    client_socket = client_socket2  # The training machine
                    gen_socket = client_socket1     # The other generating machine
            else:
                msg1 = JanggiCoach.recvSocket(client_socket1)
                assert msg1 == 'RecvProc'
                client_socket = client_socket1
        
        # If lab machine, only create a client socket
        else:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))	
            log.info('Socket connected to host')
            JanggiCoach.sendSocket(client_socket, 'SendProc')

        # Set socket timeout
        client_socket.settimeout(2.0)
        if urp_available:
            gen_socket.settimeout(2.0)

        while True:	
            # Receive a generated data	
            data = dataQ.get()	

            # Send the data through socket
            JanggiCoach.sendSocket(client_socket, data)

            # Check if any data arrived from lab machine, and forward it
            if urp_available:
                while True:
                    data = JanggiCoach.recvSocket(gen_socket)
                    if data == None:
                        break
                    JanggiCoach.sendSocket(client_socket, data)

            # Check if any state_dict name arrived through the socket	
            data = JanggiCoach.recvSocket(client_socket)
            if data == None:
                continue

            checkpoint_name = data
            SDQ.put(checkpoint_name)
            if urp_available:
                JanggiCoach.sendSocket(gen_socket, checkpoint_name)

        # Close the socket	
        client_socket.close()
        server_socket.close()

    @staticmethod	
    def remoteRecvProcess(rrProcArgs):
        """	
        	
        """	
        cnt = 0
        dataQ, SDQ, HOST, PORT = rrProcArgs	

        # Create a socket object	
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server	
        client_socket.connect((HOST, PORT))	
        log.info('Socket connected to host')

        # Alert the haedong server
        JanggiCoach.sendSocket(client_socket, 'RecvProc')

        client_socket.settimeout (7200)

        while True:
            # Receive a data point
            data = JanggiCoach.recvSocket(client_socket)
            if data != None:
                dataQ.put(data)
                cnt += 1

            # Check if any state_dict name arrived, and send it over the socket	
            if not SDQ.empty():
                sd = SDQ.get()	
                while not SDQ.empty():	
                    sd = SDQ.get()
                JanggiCoach.sendSocket(client_socket, sd)

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
        if self.args.remote_send:
            self.learn_generate()
        else:
            self.learn_learn()

    def learn_generate(self):
        """
        Process that continuously generates self-play data
        """
        manager = mp.Manager()
        sharedQ = manager.Queue()

        # Create the server-communicating process
        remoteDataQ = manager.Queue()
        remoteSDQ = manager.Queue()
        HOST, PORT = self.args.haedong_params       
        rrProc = mp.Process(target=JanggiCoach.remoteSendProcess, args=((remoteDataQ, remoteSDQ, HOST, PORT, self.args.is_haedong),))	
        rrProc.daemon = True
        rrProc.start()

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
        for j in range(self.args.num_selfplay_procs):
            selfplayPool.apply_async(JanggiCoach.executeEpisode, [(Game(self.game.c1, self.game.c2), self.args, sharedQ, queues[j], j, nextSelfplayQ)])
        
        # Continuously generate self-plays
        while True:
            # Check for any network updates
            if not remoteSDQ.empty():
                cp_name = remoteSDQ.get()
                while not remoteSDQ.empty():
                    cp_name = remoteSDQ.get()
                for q in nn_update_pipes2:
                    q.send(cp_name)
                    q.recv()
                log.info('Alerted the nn procs to update the network')

            # Wait for a selfplay result
            data, finished_id = nextSelfplayQ.get()
            self.selfPlaysPlayed += 1
            log.info(str(self.selfPlaysPlayed)+' selfplay games played')
            remoteDataQ.put(data)

            # Run new selfplay
            selfplayPool.apply_async(JanggiCoach.executeEpisode, [(Game(self.game.c1, self.game.c2), self.args, sharedQ, queues[finished_id], finished_id, nextSelfplayQ)])

    def learn_learn(self):
        """
        Process that generates numEps self-play data, and trains the network
        """
        manager = mp.Manager()	
        sharedQ = manager.Queue()	

        # Create the server-communicating processes
        HOST, PORT = self.args.haedong_params
        remoteDataQ = manager.Queue()
        remoteSDQ = manager.Queue()
        rrProc = mp.Process(target=JanggiCoach.remoteRecvProcess, args=((remoteDataQ, remoteSDQ, HOST, PORT),))	
        rrProc.daemon = True	
        rrProc.start()

        # Generate self-plays and train

        for i in range(1, self.args.numIters + 1):
            # Create numEps queues
            queues = []
            for j in range(self.args.numEps):
                queues.append(manager.Queue())

            # Create num_gpu_procs nnProcess
            nnProcs = []
            for j in range(self.args.num_gpu_procs):
                # Run nnProc	
                state_dict = {k: v.cpu() for k, v in self.nnet.nnet.state_dict().items()}
                nnProc = mp.Process(target=JanggiCoach.nnProcess, args=[(self.game, state_dict, sharedQ, self.args.gpus_to_use[j%len(self.args.gpus_to_use)], queues, None)])	
                nnProc.daemon = True
                nnProc.start()
                nnProcs.append(nnProc)

            # Create self-play process pool
            selfplayPool = Pool(None)

            # Create pool args	
            pArgs = []	
            for j in range(self.args.numEps):
                pArgs.append((Game(self.game.c1, self.game.c2), self.args, sharedQ, queues[j], j, None))

            # bookkeeping
            log.info(f'Starting Iter #{i} ... ({self.selfPlaysPlayed} games played)')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                log.info('Start generating self-plays')	
                with tqdm(total = self.args.numEps) as pbar:	
                    for d in tqdm(selfplayPool.imap_unordered(JanggiCoach.executeEpisode, pArgs)):	
                        if self.args.remote_send:	
                            remoteDataQ.put(d)	
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
            self.saveTrainExamples(self.selfPlaysPlayed)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            log.info('TRAINING AND SAVING NEW MODEL')	
            self.nnet.train(trainExamples)	
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(self.selfPlaysPlayed))
            
            for f in self.args.checkpoint_folders:
                log.info('SENDING CHECKPOINT TO SERVER')
                with open(self.getStateDictFile(f, self.selfPlaysPlayed), 'wb') as handle:
                    pickle.dump({k: v.cpu() for k, v in self.nnet.nnet.state_dict().items()}, handle)

            # Send the new state_dict
            remoteSDQ.put(self.getStateDictFile(self.args.checkpoint_folder, self.selfPlaysPlayed))	
            log.info('Alerted updated network')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pickle'

    def getStateDictFile(self, folder, iteration):
        return os.path.join(folder, 'sd_' + str(iteration) + '.pickle')

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
