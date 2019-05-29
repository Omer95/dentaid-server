"""
Dentaid Server version 1.0.0 
Author: Omer Farooq Ahmed, Hassan Saeed

"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from PIL import Image
from keras.models import load_model
import keras
import tensorflow as tf

from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
import os
import numpy as np
import cv2
import firebase_admin
from firebase_admin import credentials, storage, db
import json
import base64
import jsonify
import requests

class InferenceConfig(Config):
    NAME = "teeth"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # BG has 80 classes
    IMAGE_SHAPE = [1991, 1127, 3]

def init():
    # Root directory of the project
    ROOT_DIR = r'C:\Users\maazh\Desktop\Server\dentaid-server-master'

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library


    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights.h5")
    # Download COCO trained weights from Releases if needed

    inference_config = InferenceConfig()
    inference_config.IMAGE_SHAPE = [1024, 1024, 3]
    inference_config.IMAGE_MAX_DIM = 1024
    #inference_config.display()
    # Create model object in inference mode.
    #keras.backend.clear_session()    
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=inference_config)
    
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    global graph
    graph = tf.get_default_graph()
    #keras.backend.clear_session()
    resnet_model = load_model('resnet50_model_tf_16bs_90ep_1340_training_400_valid_sgd.h5')
    return model, resnet_model

def checkCary(teethSet):
    teethSet = np.array(teethSet)
    return teethSet
    print(teethSet.shape)


#init app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model, resnet_model = init()
# init firebase admin and storage bucket
cred = credentials.Certificate('./dentaid-diagnostics-firebase-adminsdk-vx454-8291a3a57e.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'dentaid-diagnostics.appspot.com',
    'databaseURL': 'https://dentaid-diagnostics.firebaseio.com/'
})
bucket = storage.bucket()

@app.route("/dentist", methods=["GET", "POST"])
@cross_origin()
def hi():
    if request.method == 'GET':
        return 'hi'
    elif request.method == 'POST':
        payload = request.get_json()
        print(payload.get('imageUrl'))
        # download image from firebase storage
        r = requests.get(payload.get('imageUrl'))
        with open('temp2.jpeg', 'wb') as f:
            f.write(r.content)
            
        img = cv2.imread('temp2.jpeg')
        
        detect("temp.jpg", model, img)

        
        # upload image to firebase storage bucket
        blob = bucket.blob(payload.get('userId') +'/'+ payload.get('filePath').split('/')[1] + '_1') 
        blob.upload_from_filename('./temp.jpg')
        print('image: ' + payload.get('filePath') + ' uploaded to firebase storage')
        print(blob.public_url)
        # set filename in realtime database
        #ref = db.reference('/users/'+payload.get('userId')+'/patients/'+)
        #ref.push({'filename': payload.get('filename')})
        
        return str(blob.public_url)
        
@app.route("/", methods=["GET","POST"])
@cross_origin()
def hello():
    #checkS()
    #return 's'
    
    #image = skimage.io.imread(r'C:\Users\maazh\Desktop\Server\dentaid-server-master\995.jpeg', as_grey=False)
    #detect('sad.jpg',model, image)
    #checkCarry
    #print('ss')
    #detect("731.jpg", model, img)
    if request.method == 'POST':
        payload = request.get_json()     
        # write the image to file
        img_str = payload.get('image')
        img_bytes = img_str.encode('utf-8')
        nparr = np.frombuffer(base64.b64decode(img_bytes), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        #cv2.imshow('POST image', img)
        #cv2.waitKey(0)
        
        detect("temp.jpg", model, img)

        #cv2.imwrite('temp.jpg', img)
        # upload image to firebase storage bucket
        blob = bucket.blob('opgclient/' + payload.get('filename')) 
        blob.upload_from_filename('./temp.jpg')
        print('image: ' + payload.get('filename') + ' uploaded to firebase storage')
        # set filename in realtime database
        ref = db.reference('/opgclient')
        ref.push({'filename': payload.get('filename')})
        return jsonify({
            "response": "Image received and uploaded to firebase"
        })

    return 'ok'
    
    
    

def detect(outputFile, model, image):
    image = skimage.color.gray2rgb(image)
    #keras.backend.clear_session()
    # Run detection
    with graph.as_default():
        results = model.detect([image], verbose=1)
    # Visualize results
    r  = results[0]
    teethSet = save_individuals(image, r['rois'], r['masks'], r['class_ids'], r['scores'])
    
    aSet = checkCary(teethSet)
    #print(aSet)
    with graph.as_default():
        preds = resnet_model.predict(aSet)
    preds_class = np.argmax(preds, axis=1)
    print(preds_class)
    carryContain = []
    for i in range(len(preds_class)):
        if preds_class[i] == 1:
            carryContain.append(i)
    visualize.display_instances(outputFile, carryContain, image, r['rois'], r['masks'], r['class_ids'], r['scores'])
    #cv2.imwrite('temp1.jpg', img)





def save_individuals(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    
    teethSet = []
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

   

    masked_image = image.astype(np.uint32).copy()
    
    #INDIVIDUAL EXTRACTION
    maxV = 0
    maxH = 0
    
    """
    for i in range(N):

        mask = masks[:, :, i]

        y1, x1, y2, x2 = boxes[i]

        if y2-y1 > maxV:
            maxV = y2-y1
        if x2-x1 > maxH:
            maxH = x2-x1
    print(maxV, maxH)
    """
    maxV = 300
    maxH = 250
    for i in range(N):

        mask = masks[:, :, i]

        y1, x1, y2, x2 = boxes[i]

        tmp = image.copy()
        tmp[mask==0] = (0,0,0)

        if y2-y1 != maxV:
            diff = maxV - (y2-y1)
            y2 = y2 + diff//2
            y1 = y1 - diff//2
        if y2-y1 != maxV:
            y2 += 1
            
        if x2-x1 != maxH:
            diff = maxH - (x2-x1)
            x2 = x2 + diff//2
            x1 = x1 - diff//2
            
        if x2-x1 != maxH:
            x2 += 1
            
        tmp = tmp[y1:y2, x1:x2] #SAVE THIS IMAGE IF YOU NEED TO SAVE. THIS REPRESENTS 1 TOOTH
        #print(len(tmp), len(tmp[0]), len(tmp[1]), len(tmp[2]))
        #print(tmp.shape)
        #UNCOMMENT BELOW LINES TO SAVE EACH TOOTH
        #im = Image.fromarray(tmp)
        #im.save(str(i) + ".jpeg")
        teethSet.append(tmp)
        
    return teethSet
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=4400)