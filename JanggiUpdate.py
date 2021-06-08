import requests, pickle
from JanggiMainConstants import request_base_url

requests.post(url = request_base_url+"/updateIBS", data = pickle.dumps(2))
#requests.get(url = request_base_url+"/releaseLock", data = pickle.dumps(2))
