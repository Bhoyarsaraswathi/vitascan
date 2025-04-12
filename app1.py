from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import image_fuzzy_clustering as fem
import label_image
from PIL import Image

app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to save image
def save_img(img, filename):
    picture_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(img)
    image.save(picture_path)
    return picture_path

# Function to process and predict
def process(image_path):
    print("[INFO] Performing image clustering...")
    fem.plot_cluster_img(image_path, 3)
    print("[INFO] Clustering completed.")
    clustered_path = os.path.join('static', 'images', 'orig_image.jpg')
    return label_image.main(clustered_path) or "Prediction failed"

# Routes
@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload')
def upload():
    return render_template('index1.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        f = request.files['file']
        original_filename = secure_filename(f.filename)
        original_path = save_img(f, original_filename)
        result = process(original_path)

        if not result or result == "Prediction failed":
            return jsonify({'error': 'Prediction failed'}), 500

        return jsonify({'prediction': result})
    except Exception as e:
        print(f"[ERROR] /upload_image failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    from video_detect import detect_best_face
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        video = request.files['file']
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        best_face_path = detect_best_face(video_path)
        if not best_face_path or not os.path.exists(best_face_path):
            return jsonify({'error': 'No face detected'}), 400

        result = process(best_face_path)
        if not result or result == "Prediction failed":
            return jsonify({'error': 'Prediction failed'}), 500

        return jsonify({'prediction': result})
    except Exception as e:
        print(f"[ERROR] /upload_video failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/record_video', methods=['POST'])
def record_video_route():
    from video_detect import detect_best_face
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        f = request.files['file']
        filename = secure_filename(f.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(video_path)

        best_face_path = detect_best_face(video_path)
        if not best_face_path or not os.path.exists(best_face_path):
            return jsonify({'error': 'No face detected'}), 400

        result = process(best_face_path)
        if not result or result == "Prediction failed":
            return jsonify({'error': 'Prediction failed'}), 500

        return jsonify({'prediction': result})
    except Exception as e:
        print(f"[ERROR] /record_video failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        i = request.form.get('cluster')
        f = request.files['file']
        original_pic_path = save_img(f, secure_filename(f.filename))
        fem.plot_cluster_img(original_pic_path, int(i))
        return render_template('success.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        result = label_image.main(file_path).title()
        os.remove(file_path) if os.path.exists(file_path) else None

        descriptions = {
            "Vitamin A": " → Deficiency of vitamin A is associated with significant morbidity and mortality...",
            "Vitamin B": " → Vitamin B12 deficiency may lead to anemia, fatigue, and nervous system damage...",
            "Vitamin C": " → Vitamin C deficiency (scurvy) causes bleeding gums, fatigue, and rash...",
            "Vitamin D": " → Vitamin D deficiency can lead to bone loss, fractures, and rickets in children...",
            "Vitamin E": " → Vitamin E deficiency can cause nerve and muscle damage, weakness, and vision issues..."
        }

        return result + descriptions.get(result, " → No additional details found.")
    return None

# Run app
if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True)
