from flask import Flask, render_template, jsonify, request, send_from_directory, Response, redirect, url_for, session, flash
import os
import smtplib
import json
import cv2
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from tensorflow.keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
from functools import wraps


# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploaded_videos'
app.config['ALERTS_FILE'] = 'alerts.json'
app.config['PREVIOUS_ACCIDENTS_FOLDER'] = 'static/accidents'
app.config['GRAPH_FOLDER'] = 'static/graphs'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREVIOUS_ACCIDENTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)

# Initialize JSON files if they don't exist
if not os.path.exists(app.config['ALERTS_FILE']):
    with open(app.config['ALERTS_FILE'], 'w') as f:
        json.dump([], f)

# SQLite database initialization
def init_sqlite_db():
    conn = sqlite3.connect('users.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT NOT NULL,
                 email TEXT NOT NULL UNIQUE,
                 password TEXT NOT NULL,
                 role TEXT NOT NULL);''')
    conn.close()

init_sqlite_db()

# User-related functions
def get_user_by_email(email):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def add_user(username, email, password, role='user'):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
                   (username, email, generate_password_hash(password), role))
    conn.commit()
    conn.close()

# Accident detection functions
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def predict_frame(frame, model):
    img = preprocess_frame(frame)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    return predicted_class, confidence

def send_email_alert(alert):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    sender_email = os.getenv('abhishekdubalgonde@gmail.com')
    receiver_email = os.getenv('techabhi.6200@gmail.com')
    password = os.getenv('tlgn fcnu xyxf ppuv')

    subject = "Accident Detected"
    body = f"An accident was detected at {datetime.fromtimestamp(alert['time']).strftime('%Y-%m-%d %H:%M:%S')}.\nLocation: {alert['location']}\n\n"
    google_maps_link = f"https://www.google.com/maps/search/?api=1&query={alert['location'].replace(' ', '+')}"
    body += f"Google Maps Directions: {google_maps_link}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(alert['photo'], 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(alert['photo'])}")
        msg.attach(part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def save_alert(alert):
    try:
        alert['viewed'] = False  # Add viewed status
        with open(app.config['ALERTS_FILE'], 'r+') as f:
          alerts = json.load(f)
          alerts.append(alert)
          f.seek(0)
          json.dump(alerts, f)
    except Exception as e:
        print(f"Failed to save alert: {e}")

# Route definitions
@app.route('/')
def index():
    if 'email' in session:
        user = get_user_by_email(session['email'])
        user_data = {'username': user[1], 'email': user[2], 'role': user[4]}
        if user:
            if user[4] == 'admin':
                return render_template('admin.html', user=user_data)
            else:
                return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'email' in session:
        user = get_user_by_email(session['email'])
        user_data = {'username': user[1], 'email': user[2], 'role': user[4]}
        
        # Fetching data for the logged-in user
        with open(app.config['ALERTS_FILE'], 'r') as f:
            alerts = json.load(f)
            user_alerts = [alert for alert in alerts if alert['email'] == user_data['email']]
        
        return render_template('home.html', user=user_data, alerts=user_alerts)
    return redirect(url_for('login'))

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value)
    return value.strftime(format)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = get_user_by_email(email)
        if user and check_password_hash(user[3], password):
            session['email'] = user[2]  # user[2] is the email field
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html')
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password == confirm_password:
            if not get_user_by_email(email):
                add_user(username, email, password)
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Email already registered', 'danger')
        else:
            flash('Passwords do not match', 'danger')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'email' not in session:
        return 'Unauthorized', 403

    file = request.files.get('video')
    if not file or file.filename == '':
        return 'No selected file', 400

    user = get_user_by_email(session['email'])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    threading.Thread(target=detect_accidents, args=(file_path, user[1], user[2])).start()

    return 'Video uploaded successfully', 200

def detect_accidents(video_path, username, email):
    cap = cv2.VideoCapture(video_path)
    model = load_model('accident_detection_model.h5')
    accident_detected = False
    accident_count = 0
    max_accident_images = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        prediction, confidence = predict_frame(frame, model)
        if prediction == 1 and confidence >= 0.9:
            if not accident_detected:
                accident_detected = True
                accident_count = 0

            if accident_count < max_accident_images:
                timestamp = time.time()
                photo_path = f'static/accidents/accident_{datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
                cv2.imwrite(photo_path, frame)
                location = '1234 Example St, City, Country'
                description = "A possible accident was detected."
                alert = {
                    'time': timestamp,
                    'location': location,
                    'photo': photo_path,
                    'description': description,
                    'username': username,
                    'email': email,
                    'viewed': False
                }
                save_alert(alert)
                send_email_alert(alert)
                accident_count += 1
        else:
            accident_detected = False

        time.sleep(1)
    cap.release()

@app.route('/graph')
def graph_page():
    if 'email' in session:
        user = get_user_by_email(session['email'])
        user_data = {'username': user[1], 'email': user[2], 'role': user[4]}
        return render_template('graphical_analysis.html', user=user_data)
    return redirect(url_for('login'))
    

@app.route('/profile')
def profile():
    if 'email' in session:
        user = get_user_by_email(session['email'])
        user_data = {'username': user[1], 'email': user[2], 'role': user[4]}
        return render_template('profile.html', user=user_data)
    return redirect(url_for('login'))

@app.route('/generate_graph')
def generate_graph():
    with open(app.config['ALERTS_FILE'], 'r') as f:
        alerts = json.load(f)

    # Extract data for the graph
    dates = [datetime.fromtimestamp(alert['time']).strftime('%Y-%m-%d') for alert in alerts]
    unique_dates = list(set(dates))
    counts = [dates.count(date) for date in unique_dates]

    # Generate the bar graph
    plt.figure(figsize=(10, 7))
    plt.bar(unique_dates, counts, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents per Day')
    plt.xticks(rotation=45)
    
    graph_path = os.path.join(app.config['GRAPH_FOLDER'], 'accidents_per_day.png')
    plt.savefig(graph_path)
    plt.close()

    return send_from_directory(app.config['GRAPH_FOLDER'], 'accidents_per_day.png')

@app.route('/notifications')
def notifications():
    def stream():
        while True:
            with open(app.config['ALERTS_FILE'], 'r') as f:
                alerts = json.load(f)
                new_alerts = [alert for alert in alerts if not alert.get('viewed', False)]
                if new_alerts:
                    yield f"data: {json.dumps(new_alerts[-1])}\n\n"
            time.sleep(5)
    return Response(stream(), mimetype='text/event-stream')

@app.route('/accident_detection', methods=['GET', 'POST'])
def accident_detection():
  if 'email' in session:
    user = get_user_by_email(session['email'])
    user_data = {'username': user[1], 'email': user[2], 'role': user[4]}
    if request.method == 'POST':
        # Mark selected accidents as done
        done_alerts = request.form.getlist('done')
        with open(app.config['ALERTS_FILE'], 'r+') as f:
            alerts = json.load(f)
            for alert in alerts:
                if alert['photo'] in done_alerts:
                    alert['viewed'] = True
            f.seek(0)
            json.dump(alerts, f)
            f.truncate()

    # Load and display accident data
    with open(app.config['ALERTS_FILE'], 'r') as f:
        accident_data = json.load(f)
    
    return render_template('accident_detection.html', accident_data=accident_data, user=user_data)
  return redirect(url_for('login'))

def update_existing_alerts():
    with open(app.config['ALERTS_FILE'], 'r+') as f:
        alerts = json.load(f)
        for alert in alerts:
            if 'viewed' not in alert:
                alert['viewed'] = False
        f.seek(0)
        json.dump(alerts, f)
        f.truncate()
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value)
    return value.strftime(format)
  

@app.route('/about')
def about():
    if 'email' in session:
      user = get_user_by_email(session['email'])
      user_data = {'username': user[1], 'email': user[2], 'role': user[4]}
      return render_template('about.html', user=user_data)
    return render_template('about.html')
  

if __name__ == "__main__":
    update_existing_alerts()
    app.run(debug=False, host='0.0.0.0')
