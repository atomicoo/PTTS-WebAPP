import os
import os.path as osp
import requests
from urllib.parse import urlencode
import json, time, uuid
import numpy as np
from scipy.io.wavfile import write


url = "http://127.0.0.1:5000"

payload = {
    "speed": 4,
    "volume": 4,
    "tone": 4,
    "text": "To install precompiled package of eSpeak NG on Linux, use standard package manager of your distribution.",
}
headers = {
    'content-type': "application/json"
}

outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)


print("="*12 + " POST TEST " + "="*12)
data = json.dumps(payload)
response = requests.request("POST", url+"/api/mytts", data=data, headers=headers)
if response.status_code == 200:
    content = response.content.decode('utf-8')
    content = json.loads(content)
    wave, sr = content['wave'], content['sr']
    print('Saving audio...')
    filename = osp.join(outputs_dir, f"{time.strftime('%Y-%m-%d')}_{uuid.uuid4()}.wav")
    write(filename, sr, np.array(wave, dtype=np.float32))
    print(f"Audios saved to {outputs_dir}. Done.")
    print("POST TEST SUCCESSED!")
else:
    print("POST TEST FAILED!")


print("="*12 + " GET  TEST " + "="*12)
data = urlencode(payload)
response = requests.request("GET", url+"/api/mytts?"+data, headers=headers)
if response.status_code == 200:
    content = response.content.decode('utf-8')
    content = json.loads(content)
    wave, sr = content['wave'], content['sr']
    print('Saving audio...')
    filename = osp.join(outputs_dir, f"{time.strftime('%Y-%m-%d')}_{uuid.uuid4()}.wav")
    write(filename, sr, np.array(wave, dtype=np.float32))
    print(f"Audios saved to {outputs_dir}. Done.")
    print("GET TEST SUCCESSED!")
else:
    print("GET TEST FAILED!")
