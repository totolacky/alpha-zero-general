import pickle
import JanggiServerConstants as JSC
from flask import Flask, request
from threading import Lock

app = Flask(__name__)

data = []
num_data = 0
state_dict = "Default"
initial_board_state = 1
lock = Lock()
performance = []

@app.route("/getData", methods=["GET"])
def get_data():
    global data, num_data, lock
    with lock:
        tmp_data = data
        tmp_num = num_data
        data = []
        num_data = 0
    res = (tmp_num, tmp_data)
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [PERF] "+str(performance))
    return pickle.dumps(res)

@app.route("/postData", methods=["POST"])
def post_data():
    global data, num_data, lock
    new_data = pickle.loads(request.data)
    with lock:
        data += new_data
        num_data += 1
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [PERF] "+str(performance))
    return "ok"

@app.route("/getPerf", methods=["GET"])
def get_data():
    global performance
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [PERF] "+str(performance))
    return pickle.dumps(performance)

@app.route("/postPerf", methods=["POST"])
def post_data():
    global performance, lock
    new_perf_tuple = pickle.loads(request.data)
    with lock:
        performance.append(new_perf_tuple)
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [PERF] "+str(performance))
    return "ok"

@app.route("/getIBS", methods=["GET"])
def get_ibs():
    global initial_board_state
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [PERF] "+str(performance))
    return pickle.dumps(initial_board_state)

@app.route("/updateIBS", methods=["POST"])
def update_ibs():
    global initial_board_state, lock
    new_ibs = pickle.loads(request.data)
    with lock:
        initial_board_state = new_ibs
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [PERF] "+str(performance))
    return "ok"

@app.route("/getSD", methods=["GET"])
def get_sd():
    global state_dict
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [PERF] "+str(performance))
    return pickle.dumps(state_dict)

@app.route("/updateSD", methods=["POST"])
def update_sd():
    global state_dict, lock
    new_sd = pickle.loads(request.data)
    with lock:
        state_dict = new_sd
    print("[Data] "+str(num_data)+" [SD] "+str(state_dict)+" [IBS] "+str(initial_board_state)+" [PERF] "+str(performance))
    return "ok"

if __name__ == "__main__": 
    app.run(debug=True, host=JSC.HOST, port=JSC.PORT)