from flask import Flask, request, jsonify, Response
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import firebase_admin
from firebase_admin import credentials, storage
import json
import base64
#init app
app = Flask(__name__)
# init firebase admin and storage bucket
cred = credentials.Certificate('./dentaid-diagnostics-firebase-adminsdk-vx454-8291a3a57e.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'dentaid-diagnostics.appspot.com'
})
bucket = storage.bucket()

@app.route("/", methods=["GET","POST"])
def hello():
    if request.method == 'POST':
        payload = request.get_json()     
        # write the image to file
        img_str = payload.get('image')
        img_bytes = img_str.encode('utf-8')
        nparr = np.frombuffer(base64.b64decode(img_bytes), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite('temp.jpg', img)
        # upload image to firebase storage bucket
        blob = bucket.blob('opgclient/' + payload.get('filename')) 
        blob.upload_from_filename('./temp.jpg')
        print('image: ' + payload.get('filename') + ' uploaded to firebase storage')
        return jsonify({
            "response": "Image received and uploaded to firebase"
        })
    

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)