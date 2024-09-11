from flask import Flask, Response, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('./accident_detection_model.h5')

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploaded_videos'
app.config['ALERTS_FILE'] = 'alerts.json'
app.config['PREVIOUS_ACCIDENTS_FOLDER'] = 'static/accidents'
app.config['GRAPH_FOLDER'] = 'static/graphs'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREVIOUS_ACCIDENTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)

def process_frame(frame):
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)
    
    # Predict using the model
    prediction = model.predict(input_frame)
    print("Prediction:", prediction)  # Debugging line

    if prediction[0][0] > 0.5:  # Adjust the threshold as per your model's output
        print("Accident detected!")
        save_detected_frame(frame)

def save_detected_frame(frame):
    """
    Save the frame to the 'dude' directory with a timestamp.
    """
    # Ensure the 'dude' directory exists
    save_dir = './dude'
    os.makedirs(save_dir, exist_ok=True)

    # Create a filename with the current timestamp
    filename = os.path.join(save_dir, f"accident_{int(time.time())}.jpg")

    # Save the frame as a JPEG image
    cv2.imwrite(filename, frame)

def get_camera_feed():
    """
    Connect to the device's camera feed, process each frame, and yield it as a response.
    """
    # 0 is typically the default camera (your laptop's built-in camera or the first connected camera)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect accident in this frame
        process_frame(frame)

        # Convert the frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/camera_feed')
def camera_feed():
    """
    Flask route to serve the video feed.
    """
    return Response(get_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """
    Render the main page with live camera feeds.
    """
    return render_template('admin.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
