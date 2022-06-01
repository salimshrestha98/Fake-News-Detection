from test_tf import test

import flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''


@app.route('/api/fnd', methods=['GET', 'POST'])
def test_fake():
    news = request.form.get('news')

    # if test.check_fake(news):
    #     response = flask.jsonify("Fake")
    # else:
    #     response = flask.jsonify("Not Fake")

    res = test.check_fake(news)
    print(res)
    response = flask.jsonify({'result': int(res)})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
app.run()