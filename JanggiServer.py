import pickle
import JanggiMainConstants as JMC
from flask import Flask, request
from threading import Lock

app = Flask(__name__)

data = []
num_data = 0
state_dict = "Default"
initial_board_state = 1
lock = Lock()
performance = []
lock_held = False
next_game = []
evaluation = []

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

@app.route("/getNextGame", methods=["GET"])
def get_next_game():
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
                # (checkpoint, is_rp, is_p1)
            else:
                assert e[2][0] != 0
                e[2][0] -= 1
                print("[GetNextGame] Next game is checkpoint "+str(e[0])+" vs. GP")
                return pickle.dumps((e[0], False, e[2][0] % 2))

@app.route("/pushEval", methods=["POST"])
def push_eval():
    global lock, next_game
    new_sd = pickle.loads(request.data)
    with lock:
        next_game.append((new_sd, JMC.eval_num, JMC.eval_num))
        # (checkpoint, rp todo, gp todo)
        evaluation.append((new_sd, (0, 0), (0, 0)))
        # (checkpoint, (rp done, rp won), (gp done, gp won))
    print("Should evaluate checkpoint " + str(new_sd))
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
    app.run(debug=True, host=JMC.HOST, port=JMC.PORT)
