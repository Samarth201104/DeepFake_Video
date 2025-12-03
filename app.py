# # from flask import Flask, render_template, request
# # import os
# # import cv2
# # import numpy as np
# # from tensorflow.keras.models import load_model
# # from retinaface import RetinaFace
# # from PIL import Image

# # app = Flask(__name__)
# # app.config['UPLOAD_FOLDER'] = 'uploads'

# # # Load trained model
# # model = load_model("xception_best.h5")

# # def extract_faces(frame, target_size=(256, 256)):
# #     """
# #     Detect and return all faces from a frame using RetinaFace.
# #     Returns a list of resized face images (numpy arrays).
# #     """
# #     faces_list = []
# #     try:
# #         detections = RetinaFace.detect_faces(frame)
# #     except Exception:
# #         return faces_list

# #     if not isinstance(detections, dict):
# #         return faces_list

# #     for face in detections.values():
# #         x1, y1, x2, y2 = face["facial_area"]
# #         x1, y1 = max(0, x1), max(0, y1)
# #         crop = frame[y1:y2, x1:x2]

# #         if crop.size == 0:
# #             continue

# #         face_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
# #         face_img = face_img.resize(target_size)
# #         faces_list.append(np.array(face_img))

# #     return faces_list

# # def predict_video(video_path, model, frame_skip=10):
# #     """
# #     Predict whether the video is real or fake using all detected faces.
# #     Returns label and confidence.
# #     """
# #     cap = cv2.VideoCapture(video_path)
# #     predictions = []

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
# #         if frame_id % frame_skip != 0:
# #             continue

# #         faces = extract_faces(frame)
# #         if not faces:
# #             print(f"[Frame {frame_id}] No faces detected")
# #             continue

# #         for idx, face in enumerate(faces, start=1):
# #             img = face / 255.0
# #             img = np.expand_dims(img, axis=0)  # (1, 256, 256, 3)
# #             pred = model.predict(img, verbose=0)
# #             prob = float(pred[0][0])
# #             predictions.append(prob)

# #             binary = 1 if prob >= 0.5 else 0
# #             print(f"[Frame {frame_id}] Face {idx}: Prediction={binary}, Probability={prob:.4f}")

# #     cap.release()

# #     if not predictions:
# #         return "No faces detected", 0.0

# #     avg_pred = np.mean(predictions)
# #     label = "Fake" if avg_pred >= 0.5 else "Real"
# #     confidence = max(avg_pred, 1 - avg_pred)
# #     return label, confidence

# # @app.route('/', methods=['GET', 'POST'])
# # def index():
# #     result = None
# #     confidence = None
# #     if request.method == 'POST':
# #         if 'video' not in request.files:
# #             return render_template('index.html', result="No file uploaded")

# #         file = request.files['video']
# #         if file.filename == '':
# #             return render_template('index.html', result="No file selected")

# #         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
# #         file.save(filepath)

# #         # Predict
# #         result, confidence = predict_video(filepath, model)
# #         confidence = round(confidence * 100, 2)

# #     return render_template('index.html', result=result, confidence=confidence)

# # if __name__ == "__main__":
# #     app.run(debug=True)


# from flask import Flask, render_template, request, send_from_directory
# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from retinaface import RetinaFace
# from PIL import Image

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'

# # Load trained model
# model = load_model("xception_best.h5")


# def extract_faces(frame, target_size=(256, 256)):
#     """
#     Detect and return all faces from a frame using RetinaFace.
#     Returns a list of resized face images (numpy arrays).
#     """
#     faces_list = []
#     try:
#         detections = RetinaFace.detect_faces(frame)
#     except Exception:
#         return faces_list

#     if not isinstance(detections, dict):
#         return faces_list

#     for face in detections.values():
#         x1, y1, x2, y2 = face["facial_area"]
#         x1, y1 = max(0, x1), max(0, y1)
#         crop = frame[y1:y2, x1:x2]

#         if crop.size == 0:
#             continue

#         face_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#         face_img = face_img.resize(target_size)
#         faces_list.append(np.array(face_img))

#     return faces_list


# def predict_video(video_path, model, frame_skip=10):
#     """
#     Predict whether the video is real or fake using all detected faces.
#     Returns label and confidence.
#     """
#     cap = cv2.VideoCapture(video_path)
#     predictions = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         if frame_id % frame_skip != 0:
#             continue

#         faces = extract_faces(frame)
#         if not faces:
#             print(f"[Frame {frame_id}] No faces detected")
#             continue

#         for idx, face in enumerate(faces, start=1):
#             img = face / 255.0
#             img = np.expand_dims(img, axis=0)  # (1, 256, 256, 3)
#             pred = model.predict(img, verbose=0)
#             prob = float(pred[0][0])
#             predictions.append(prob)

#             binary = 1 if prob >= 0.5 else 0
#             print(f"[Frame {frame_id}] Face {idx}: Prediction={binary}, Probability={prob:.4f}")

#     cap.release()

#     if not predictions:
#         return "No faces detected", 0.0

#     avg_pred = np.mean(predictions)
#     label = "Fake" if avg_pred >= 0.5 else "Real"
#     confidence = max(avg_pred, 1 - avg_pred)
#     return label, confidence


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = None
#     confidence = None
#     video_path = None

#     if request.method == 'POST':
#         if 'video' not in request.files:
#             return render_template('index.html', result="No file uploaded")

#         file = request.files['video']
#         if file.filename == '':
#             return render_template('index.html', result="No file selected")

#         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         # Predict
#         result, confidence = predict_video(filepath, model)
#         confidence = round(confidence * 100, 2)

#         # Pass video path for display
#         video_path = f"/uploads/{file.filename}"

#     return render_template('index.html', result=result, confidence=confidence, video_path=video_path)


# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request, send_from_directory, jsonify, Response
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from retinaface import RetinaFace
from PIL import Image
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your trained DeepFake model
model = load_model("xception_best.h5")

# ----------------------- Helper: Extract Faces ----------------------- #
def extract_faces(frame, target_size=(256, 256)):
    faces_list = []
    try:
        detections = RetinaFace.detect_faces(frame)
    except Exception:
        return faces_list

    if not isinstance(detections, dict):
        return faces_list

    for face in detections.values():
        x1, y1, x2, y2 = face["facial_area"]
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        face_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        face_img = face_img.resize(target_size)
        faces_list.append(np.array(face_img))
    return faces_list


# ----------------------- Route: Home ----------------------- #
@app.route('/')
def index():
    return render_template('index.html')


# ----------------------- Route: Upload ----------------------- #
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    video_path = f"/uploads/{file.filename}"
    return jsonify({'video_path': video_path})


# ----------------------- Route: Stream Predictions ----------------------- #
@app.route('/predict_stream')
def predict_stream():
    video_path = request.args.get('video', '').replace('/uploads/', '')
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], video_path)

    if not os.path.exists(full_path):
        return Response("Video not found", status=404)

    def generate():
        cap = cv2.VideoCapture(full_path)
        frame_skip = 10
        predictions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_id % frame_skip != 0:
                continue

            faces = extract_faces(frame)
            if not faces:
                yield f"data: [Frame {frame_id}] No faces detected\n\n"
                continue

            for idx, face in enumerate(faces, start=1):
                img = face / 255.0
                img = np.expand_dims(img, axis=0)
                prob = float(model.predict(img, verbose=0)[0][0])
                binary = 1 if prob >= 0.5 else 0
                predictions.append(prob)
                yield f"data: [Frame {frame_id}] Face {idx}: Prediction={binary}, Probability={prob:.4f}\n\n"
                time.sleep(0.05)  # small delay for smooth streaming

        cap.release()

        if predictions:
            avg_pred = np.mean(predictions)
            label = "Fake" if avg_pred >= 0.5 else "Real"
            confidence = round(max(avg_pred, 1 - avg_pred) * 100, 2)
            yield f"data: --- FINAL RESULT: {label} ({confidence}%) ---\n\n"

        yield "data: DONE\n\n"

    return Response(generate(), mimetype='text/event-stream')


# ----------------------- Serve Uploaded Videos ----------------------- #
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ----------------------- Run App ----------------------- #
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
