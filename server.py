import os.path as osp
import requests

from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS

from random import randint
from backend.mytts import MyTTS
from scipy.io.wavfile import write

tts = MyTTS(device='cpu')

app = Flask(__name__,
            static_folder = "./dist/static",
            template_folder = "./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if app.debug:
        return requests.get('http://127.0.0.1:8080/{}'.format(path)).text
    return render_template("index.html")


@app.route('/api/random')
def api_random():
    response = {
        'randomNumber': randint(1, 100)
    }
    return jsonify(response)

@app.route('/api/mytts', methods=['GET', 'POST'])
def api_mytts():
    req = request.json if request.method == 'POST' else request.args
    print(req)
    text, speed, volume, tone = \
        req.get('text'), req.get('speed', 4), req.get('volume', 4), req.get('tone', 4)
    waves, sr = tts([text], int(speed), int(volume), int(tone))
    filepath = osp.join('dist', 'demo.wav')
    write(filepath, sr, waves[0])
    return send_file(filepath)
    # return jsonify({'wave': waves[0].tolist(), 'sr': sr})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
