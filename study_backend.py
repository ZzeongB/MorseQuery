"""
Flask backend for user study data collection.
Stores participant data server-side instead of relying on browser downloads.
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import json
import os
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# Data directory
DATA_DIR = 'study_data'
os.makedirs(DATA_DIR, exist_ok=True)

# Base directory for serving files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    """Serve Page 1: Data Collection"""
    return send_from_directory(BASE_DIR, 'study_page1_backend.html')


@app.route('/label')
def label():
    """Serve Page 2: Retrospective Labeling"""
    return send_from_directory(BASE_DIR, 'study_page2_backend.html')


@app.route('/video/<path:filename>')
def video(filename):
    """Serve video files with proper MIME type"""
    video_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    return send_file(video_path, mimetype='video/mp4')


@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Create new participant session"""
    session_id = str(uuid.uuid4())
    session_data = {
        'session_id': session_id,
        'created_at': datetime.now().isoformat(),
        'gestures': [],
        'labeled_data': []
    }

    # Save session
    filepath = os.path.join(DATA_DIR, f'{session_id}.json')
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2)

    return jsonify({'session_id': session_id})


@app.route('/api/gesture/log', methods=['POST'])
def log_gesture():
    """Log a gesture timestamp (Page 1)"""
    data = request.json
    session_id = data.get('session_id')
    gesture = {
        'gesture_timestamp': data.get('gesture_timestamp'),
        'video_time': data.get('video_time')
    }

    # Load session
    filepath = os.path.join(DATA_DIR, f'{session_id}.json')
    with open(filepath, 'r') as f:
        session_data = json.load(f)

    # Add gesture
    session_data['gestures'].append(gesture)

    # Save session
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2)

    return jsonify({'success': True})


@app.route('/api/gestures/get/<session_id>', methods=['GET'])
def get_gestures(session_id):
    """Get all gestures for a session (Page 2)"""
    filepath = os.path.join(DATA_DIR, f'{session_id}.json')

    if not os.path.exists(filepath):
        return jsonify({'error': 'Session not found'}), 404

    with open(filepath, 'r') as f:
        session_data = json.load(f)

    return jsonify({'gestures': session_data['gestures']})


@app.route('/api/label/submit', methods=['POST'])
def submit_label():
    """Submit labeled data for a gesture (Page 2)"""
    data = request.json
    session_id = data.get('session_id')

    # Load session
    filepath = os.path.join(DATA_DIR, f'{session_id}.json')
    with open(filepath, 'r') as f:
        session_data = json.load(f)

    # Add labeled data
    labeled_gesture = {
        'gesture_timestamp': data.get('gesture_timestamp'),
        'video_time': data.get('video_time'),
        'intended_action': data.get('intended_action'),
        'intended_action_other': data.get('intended_action_other'),
        'target_type': data.get('target_type'),
        'selected_text': data.get('selected_text'),
        'target_description': data.get('target_description'),
        'target_other': data.get('target_other'),
        'desired_output_form': data.get('desired_output_form'),
        'desired_output_other': data.get('desired_output_other')
    }

    session_data['labeled_data'].append(labeled_gesture)
    session_data['updated_at'] = datetime.now().isoformat()

    # Save session
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2)

    return jsonify({'success': True})


@app.route('/api/session/complete/<session_id>', methods=['POST'])
def complete_session(session_id):
    """Mark session as complete and return final data"""
    filepath = os.path.join(DATA_DIR, f'{session_id}.json')

    if not os.path.exists(filepath):
        return jsonify({'error': 'Session not found'}), 404

    with open(filepath, 'r') as f:
        session_data = json.load(f)

    session_data['completed_at'] = datetime.now().isoformat()

    # Save final session
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2)

    return jsonify({'labeled_data': session_data['labeled_data']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
