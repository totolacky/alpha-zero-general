import json
import JanggiServerConstants as JSC
from flask import Flask, request
from threading import Lock

app = Flask(__name__)

data = []
num_data = 0
state_dict = ""
initial_board_state = 0
lock = Lock()

@app.route("/getData", methods=["GET"])
def get_data():
    global data, num_data, lock
    with lock:
        tmp_data = data
        tmp_num = num_data
        data = []
        num_data = 0
    res = (tmp_num, tmp_data)
    print("Sent "+str(tmp_num)+" selfplay data")
    return json.dumps(res)

@app.route("/postData", methods=["POST"])
def post_data():
    global data, num_data, lock
    print("Received selfplay data")
    new_data = json.loads(request.get_data())
    with lock:
        data += new_data
        num_data += 1
    return "ok"

@app.route("/getIBS", methods=["GET"])
def get_ibs():
    global initial_board_state
    print("Sent IBS "+str(initial_board_state))
    return json.dumps(initial_board_state)

@app.route("/updateIBS", methods=["POST"])
def update_ibs():
    global initial_board_state, lock
    new_ibs = json.loads(request.get_data())
    with lock:
        initial_board_state = new_ibs
    print("Updated IBS to "+str(initial_board_state))
    return "ok"

@app.route("/getSD", methods=["GET"])
def get_sd():
    global state_dict
    print("Sent SD "+str(state_dict))
    return json.dumps(state_dict)

@app.route("/updateSD", methods=["POST"])
def update_sd():
    global state_dict, lock
    new_sd = json.loads(request.get_data())
    with lock:
        state_dict = new_sd
    print("Updated SD to "+str(new_sd))
    return "ok"

if __name__ == "__main__": 
    app.run(debug=True, host=JSC.HOST, port=JSC.PORT)