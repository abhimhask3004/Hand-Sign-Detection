import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from flask import Flask, render_template, Response,jsonify

app = Flask(__name__)

# Initialize OpenCV video capture, HandDetector, and Classifier
cap = None
detector = None
classifier = None
offset = 20
imgSize = 300
labels = ["0","1","2","3","4","5","6","7","8","9","A", "B", "C", "D","E","F","G","H", "Hi how are you ?","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

video_feed_active = False


def generate_frames():
    while True:
        if video_feed_active:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                   # new_label = labels[index]

                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset),
                                  (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                                (255, 255, 255),
                                2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset),
                                  (255, 0, 255),
                                  4)
                    # ans = "".join(labels[index])
                    # print(ans)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset),
                                  (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                                (255, 255, 255),
                                2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset),
                                  (255, 0, 255),
                                  4)
                # ans = "".join(labels[index])
                # print(ans)
                # new_label=labels[index]

                # cv2.imshow("imagecrop", imgCrop)
                # cv2.imshow("imageWhite", imgWhite)
                cv2.imshow("image", imgOutput)

                #prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Convert the frame to JPEG format for HTML display
            _, buffer = cv2.imencode('.jpg', imgOutput)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('sam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video')
def start_video():
    global video_feed_active
    video_feed_active=True
    return ("video feed started")




@app.route('/stop_video')
def stop_video():
    global video_feed_active
    video_feed_active = False

    return "video feed stopped"

@app.route('/exit_video')
def exit_video():
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')



if __name__ == "__main__":
    cap=cv2.VideoCapture(0)
    detector=HandDetector(maxHands=1)
    classifier=Classifier("model/keras_model.h5", "model/labels.txt")
    app.run(debug=True)