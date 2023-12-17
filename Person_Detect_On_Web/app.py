from flask import Flask, render_template, Response
from threading import Condition
import io
import logging
import numpy as np
import cv2
from PIL import Image

import libcamera

from picamera2 import MappedArray, Picamera2, Preview
from picamera2.encoders import JpegEncoder, H264Encoder
from picamera2.outputs import FileOutput

app = Flask(__name__)
picam2 = Picamera2()
picam2_config = picam2.create_video_configuration(main={"format": "XRGB8888", "size": (640, 480)})
picam2_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
picam2.configure(picam2_config)


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


def detect_person(image_bytes):
    # Convert bytes data to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Convert PIL Image to OpenCV format
    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

    # Load pre-trained person detection model (e.g., Haar cascade or HOG + SVM)
    # Replace 'path/to/haarcascade_frontalface_default.xml' with the actual path to the model file
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect persons in the image
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    persons = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50))

    # Process detection results
    for (x, y, w, h) in persons:
        # Draw bounding box around the person
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert processed image to bytes
    _, image_buffer = cv2.imencode('.jpg', cv_image)
    image_bytes = image_buffer.tobytes()

    return image_bytes


def gen_image():
    try:
        output = StreamingOutput()
        picam2.start_recording(JpegEncoder(), FileOutput(output))
        while True:
            with output.condition:
                output.condition.wait()
                frame = output.frame
            frame = detect_person(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytes(frame) + b'\r\n')
    except Exception as ex:
        logging.warning('Exception is :  %s', str(ex))
    finally:
        picam2.stop_recording()


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/index')
def index():
    """Video streaming"""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, threaded=True)
