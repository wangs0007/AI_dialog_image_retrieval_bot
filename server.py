# -*- coding: utf-8 -*-
from flask import Flask, request
app = Flask(__name__)

@app.route('/register', methods=['POST'])
def register():
     print(request.data)
      #do something else # #
     return 'welcome'


print(request.data)
app.run(host='0.0.0.0',port=8555,debug=True)