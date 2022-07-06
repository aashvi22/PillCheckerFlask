import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
# import tensorflow as tf
# from tensorflow import keras
import numpy as np
from PIL import Image
import cv2 as cv
import numpy

from flask import Flask, request, jsonify, render_template

from PIL import Image
import pytesseract


# start block

from flask import Response
import datetime, time
import sys
from threading import Thread

global capture, rec_frame, grey, switch, neg, face, rec, out, capture_prescription
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
capture_prescription = 0

canRenderTemplate = False

global result, result_prescription
result = "nothing"
result_prescription = "no text yet"

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

camera = cv.VideoCapture(0)

def ocr_core(filename):
    text = pytesseract.image_to_string(Image.open(filename))
    return text

def gen_frames():  # generate frame by frame from camera
    print('in gen_frames')
    global out, capture, rec_frame, result, capture_prescription, result_prescription
    while True:
        success, frame = camera.read()
        if success:
            # if(face):                
            #     frame= detect_face(frame)
            if (grey):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if (neg):
                frame = cv.bitwise_not(frame)
            if (capture):
                # <class 'numpy.ndarray'>
                print('in gen_frames capture')
                result = passIntoModel(frame)
                capture = 0
                # now = datetime.datetime.now()
                # p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                # cv.imwrite(p, frame)

            if (capture_prescription):
                print('in gen_frames capture prescription')
                result_prescription = passTextIntoModel(frame)
                print(result_prescription)
                capture_prescription = 0


            if (rec):
                rec_frame = frame
                frame = cv.putText(cv.flip(frame, 1), "Recording...", (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                   4)
                frame = cv.flip(frame, 1)

            try:
                ret, buffer = cv.imencode('.jpg', cv.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


def record(out):
    global rec_frame
    while (rec):
        time.sleep(0.05)
        out.write(rec_frame)


# end block

# model = keras.models.load_model("nn2.h5")


# def passIntoModel(filearray):
#     #img = cv.imread('r{}'.format(filename))
#     # print("testing")
#     #note: filename is not type string. flask converted it to type filestorage so you'll have to somehow convert to 
#     #type string for imread. idk how lol but google it for later

#     # img = cv.imread(r'C:\Users\lim teck sin\Desktop\PillCheckerFlask\multivitaminpill.jpg')
#     # print(type(img))
#     img = cv.resize(filearray,(180,180))
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     img = img[numpy.newaxis,:,:]
#     # cv.imshow('image',img)
#     # cv.waitKey(0)
#     #problem: thinks almost everything is lacto5
#     #potential issues: 
#     # 1) cropping images weirdly 
#     # 2)math error that makes it choose first index in class names 
#     # 3)bad model

#     prediction = model.predict(img)
#     class_names = ['Lacto5', 'Multivitamin', 'Pill1', 'Pill2', 'Pill3', 'Pill4', 'Pill5', 'Pill6', 'VitaminC']
#     answer = class_names[np.argmax(prediction[0])]
#     print(answer)
#     return answer

def passTextIntoModel(filearray):
    img = Image.fromarray(filearray, 'RGB')
    prescription = pytesseract.image_to_string(img)
    return prescription


def passIntoModel(filearray):

    # img = filearray
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # threshold, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
    # blurred = cv.GaussianBlur(thresh, (15, 15), 0)
    # canny = cv.Canny(blurred, 0, 300)
    # (cnts, _) = cv.findContours(canny.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # coins = img.copy()
    # counter = 0
    # class_names = ['Blue-White Capsule', 'Multivitamin', 'Vitamin C', 'White Capsule']
    # index = 0
    # for cnt in cnts:
    #     x, y, w, h = cv.boundingRect(cnt)
    #     cv.rectangle(coins, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     blank = np.zeros(img.shape[:2], dtype='uint8')
    #     mask = cv.rectangle(blank, (x, y), (x + w, y + h), 255, -1)
    #     masked = cv.bitwise_and(img, img, mask=mask)
    #     # cv.imwrite('outputPicName' + str(counter) + '.jpg', masked) #obtain output pics which are the pics with individual pills and the pics are named with this
    #     # counter = counter + 1
    #     img6 = cv.resize(masked, (180, 180))
    #     img6 = cv.cvtColor(img6, cv.COLOR_BGR2RGB)
    #     img6 = img6[numpy.newaxis, :, :]
    #     prediction = model.predict(img6)
    #     answer = class_names[np.argmax(prediction[0])]
    #     print(index)
    #     print(answer)
    #     index = index + 1


    # prediction = model.predict(img)
    # class_names = ['Lacto5', 'Multivitamin', 'Pill1', 'Pill2', 'Pill3', 'Pill4', 'Pill5', 'Pill6', 'VitaminC']
    # answer = class_names[np.argmax(prediction[0])]
    # print(answer)
    # return answer

    filler_result = ['fake pill 1', 'fake pill 2']
    return filler_result


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print('in / directory post')
        return passIntoModel(request.files.get('file'))
    print('in / directory')
    return render_template('webcam.html')

    return "OK"


@app.route('/video_feed')
def video_feed():
    print('video feed')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    print('tasks')
    global switch, camera, result, result_prescription
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
            print('requests post about to call gen_frames')
            gen_frames()
        elif request.form.get('click') == 'Capture_Prescription':
            global capture_prescription
            capture_prescription = 1
            print('requests post prescription about to call gen_frames')
            gen_frames()
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv.destroyAllWindows()

            else:
                camera = cv.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if (rec):
                now = datetime.datetime.now()
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                out = cv.VideoWriter('vid_{}.avi'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                out.release()


    elif request.method == 'GET':

        print('returning get requests')
        print('GET prescription results are {}'.format(result_prescription))
        return render_template('webcam.html', results=result, results_prescription=result_prescription)

    print('returning requests')
    print('GET prescription results are {}'.format(result_prescription))
    return render_template('webcam.html', results=result, results_prescription=result_prescription)


if __name__ == "__main__":
    app.run(debug=True)

    # issue: printing out the last pill, not the current one
    # issue: calling returning requests before gen_frames is called. this is why pill label not updated
    # try: render template in gen frames?

    # issue:
    # thinks blue white capsule is white capsule
    # thinks white capsule is multivitamin
    # good:
    # thinks vitamin c is vitamin c
    # thinks multivitamin is multivitamin

    # issue: thinks daylight is pill

    # to do: add more to training set
    # create boolean to deal with threading
    # add other pills (1-6 and lacto 5)
