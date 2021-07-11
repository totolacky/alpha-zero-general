import pickle
import JanggiServerConstants as JMC
from flask import Flask, request
from threading import Lock

app = Flask(__name__)

data = []
num_data = 0
state_dict = "Default"
last_checkpoint = "Default"
checkpoints_to_remove = []
checkpoint_remove_buffer = []
initial_board_state = 1
lock = Lock()
performance = []
lock_held = False
next_game = []
evaluation = []

######################
#   Self-play Data   #
######################
@app.route("/getData", methods=["GET"])
def get_data():
    global data, num_data, lock
    with lock:
        tmp_data = data
        tmp_num = num_data
        data = []
        num_data = 0
    res = (tmp_num, tmp_data)
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return pickle.dumps(res)

@app.route("/postData", methods=["POST"])
def post_data():
    global data, num_data, lock
    new_data = pickle.loads(request.data)
    with lock:
        data += new_data
        num_data += 1
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return "ok"

####################
#   Game to Play   #
####################
@app.route("/getIBS", methods=["GET"])
def get_ibs():
    global initial_board_state
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return pickle.dumps(initial_board_state)

@app.route("/updateIBS", methods=["POST"])
def update_ibs():
    global initial_board_state, lock
    new_ibs = pickle.loads(request.data)
    with lock:
        initial_board_state = new_ibs
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return "ok"

@app.route("/getNextGame", methods=["GET"])
def get_next_game():
    '''
    None: self-play
    (checkpoint, is_rp, is_p1)
    '''
    global lock, next_game
    with lock:
        if (len(next_game) == 0):
            print("[GetNextGame] Next game is an ordinary game")
            return pickle.dumps(None)
        else:
            e = next_game[0]
            if (e[1][0] != 0):
                e[1][0] -= 1
                print("[GetNextGame] Next game is checkpoint "+str(e[0])+" vs. RP")
                return pickle.dumps((e[0], True, e[1][0] % 2))
            else:
                assert e[2][0] != 0
                e[2][0] -= 1
                print("[GetNextGame] Next game is checkpoint "+str(e[0])+" vs. GP")
                return pickle.dumps((e[0], False, e[2][0] % 2))

#######################
#   State_dict Info   #
#######################
@app.route("/getSD", methods=["GET"])
def get_sd():
    global state_dict
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return pickle.dumps(state_dict)

@app.route("/updateSD", methods=["POST"])
def update_sd():
    global state_dict, lock
    new_sd = pickle.loads(request.data)
    with lock:
        state_dict = new_sd
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return "ok"

@app.route("/updateSD", methods=["POST"])
def update_sds():
    global state_dict, lock
    new_sd = pickle.loads(request.data)
    with lock:
        state_dict = new_sd
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return "ok"

#########################
#   Checkpoint Backup   #
#########################
@app.route("/getLastCheckpoint", methods=["GET"])
def get_last_checkpoint():
    global last_checkpoint
    print("[GetLastCheckpoint] Last checkpoint is "+str(last_checkpoint))
    return pickle.dumps(last_checkpoint)

@app.route("/updateLastCheckpoint", methods=["POST"])
def update_last_checkpoint():
    global last_checkpoint, lock
    with lock:
        last_checkpoint = pickle.loads(request.data)
        print("[UpdateLastCheckpoint] Last checkpoint is now "+str(last_checkpoint))
    return "ok"

@app.route("/getCheckpointsToRemove", methods=["GET"])
def get_checkpoints_to_remove():
    global checkpoints_to_remove
    print("[GetCheckpointToRemove] You should remove checkpoints "+str(checkpoints_to_remove))
    return pickle.dumps(checkpoints_to_remove)

@app.route("/alertBackupDone", methods=["POST"])
def alert_backup_done():
    global checkpoints_to_remove, lock, checkpoint_remove_buffer
    with lock:
        cp = pickle.loads(request.data)
        if (cp in checkpoint_remove_buffer):
            checkpoints_to_remove.append(cp)
            if (len(checkpoints_to_remove) > JMC.max_remove_buffer_length):
                checkpoints_to_remove.pop(0)
        else:
            checkpoint_remove_buffer.append(cp)
        print("[AlertBackupDone] Backup of checkpoint "+str(checkpoints_to_remove)+" is done")
    return "ok"

############
#   Lock   #
############
@app.route("/acquireLock", methods=["GET"])
def acquire_lock():
    global lock, lock_held
    res = False
    with lock:
        res = not lock_held
        lock_held = True
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return pickle.dumps(res)

@app.route("/releaseLock", methods=["GET"])
def release_lock():
    global lock, lock_held
    with lock:
        lock_held = False
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return pickle.dumps(True)

##################
#   Evaluation   #
##################
@app.route("/getPerf", methods=["GET"])
def get_perf():
    global performance
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return pickle.dumps(performance)

@app.route("/postPerf", methods=["POST"])
def post_perf():
    global performance, lock
    new_perf_tuple = pickle.loads(request.data)
    with lock:
        performance.append(new_perf_tuple)
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [LOCK] "+str(lock_held)+" [PERF] "+str(performance))
    return "ok"

@app.route("/pushEval", methods=["POST"])
def push_eval():
    global lock, next_game, checkpoint_remove_buffer, checkpoints_to_remove
    should_eval, new_sd = pickle.loads(request.data)
    with lock:
        if (should_eval):
            next_game.append((new_sd, JMC.eval_num, JMC.eval_num))
            # (checkpoint, rp todo, gp todo)
            evaluation.append((new_sd, (0, 0), (0, 0)))
            # (checkpoint, (rp done, rp won), (gp done, gp won))
            print("[PushEval] Should evaluate checkpoint " + str(new_sd))
        else:
            if (new_sd in checkpoint_remove_buffer):
                checkpoints_to_remove.append(new_sd)
                if (len(checkpoints_to_remove) > JMC.max_remove_buffer_length):
                    checkpoints_to_remove.pop(0)
            else:
                checkpoint_remove_buffer.append(new_sd)
            print("[PushEval] No need to evaluate checkpoint " + str(new_sd))
    
    return "ok"

@app.route("/uploadEvalRes", methods=["POST"])
def upload_eval_res():
    global lock, evaluation, performance
    (sd_name, is_rp, did_win) = pickle.loads(request.data)
    with lock:
        for i in range(len(evaluation)):
            e = evaluation[i]
            if (e[0] == sd_name):
                if is_rp:
                    e[1][0] += 1
                    e[1][1] += 1 if did_win else 0
                else:
                    e[2][0] += 1
                    e[2][1] += 1 if did_win else 0
                print("[UploadEvalRes] Uploaded eval res ("+str(sd_name)+", "+str(is_rp)+", "+str(did_win)+")")
                
                # Check if evaluation for a checkpoint is done
                if (e[1][0] == JMC.eval_num and e[2][0] == JMC.eval_num):
                    performance.append((sd_name, e[1][1]/JMC.eval_num, e[2][1]/JMC.eval_num))
                    evaluation.pop(i)
                
                break
    return "ok"

if __name__ == "__main__": 
    app.run(debug=False, host=JMC.HOST, port=JMC.PORT)
