from flask import Flask, request, jsonify, Response
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import firebase_admin
from firebase_admin import credentials, storage
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
        # convert string of image data to uint8
        nparr = np.frombuffer(request.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # to-do: run test image through
        # save the image to local directory
        cv2.imwrite('temp.jpg', img)
        # upload image to firebase storage bucket
        blob = bucket.blob('new_image.jpg') 
        blob.upload_from_filename('./temp.jpg')
        print(img)
        return jsonify({
            "response": "Image received and uploaded to firebase"
        })
    

if __name__ == '__main__':
    app.run(debug=True)