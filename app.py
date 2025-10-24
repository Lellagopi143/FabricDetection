from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# Folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ANNOTATED_FOLDER'] = 'static/annotated'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATED_FOLDER'], exist_ok=True)

# Load trained YOLOv8 model
model = YOLO("best.pt")

# Color coding for classes
CLASS_COLORS = {
    "defect free": (0, 255, 0),
    "horizontal": (0, 0, 255),
    "lines": (255, 0, 0),
    "Vertical": (255, 165, 0),
    "hole": (128, 0, 128),
    "stain": (0, 255, 255)
}

@app.route('/', methods=['GET', 'POST'])
def index():
    results_data = []

    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files:
            return "No files uploaded"
        
        for file in files:
            if file.filename == '':
                continue
            
            # Save uploaded image
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # YOLO prediction
            results_list = model(file_path)
            results = results_list[0]
            
            # Annotate image
            img = cv2.imread(file_path)
            top5_indices = results.probs.top5
            top5_conf = results.probs.top5conf
            
            # Create top-5 labels
            label_lines = []
            for idx, conf in zip(top5_indices, top5_conf):
                class_name = model.names[idx]
                label_lines.append(f"{class_name}: {conf:.2f}")
            
            # Draw top-1 label on image
            top1_class = model.names[results.probs.top1]
            top1_conf = results.probs.top1conf.item()
            color = CLASS_COLORS.get(top1_class, (255, 255, 255))
            cv2.putText(img, f"{top1_class}: {top1_conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], file.filename)
            cv2.imwrite(annotated_path, img)
            
            results_data.append({
                "filename": file.filename,
                "annotated_path": annotated_path,
                "top5_labels": label_lines
            })
    
    return render_template('index.html', results=results_data)

@app.route('/uploads/<filename>')
def send_file(filename):
    return redirect(url_for('static', filename=f'annotated/{filename}'), code=301)

if __name__ == "__main__":
    app.run(debug=True)
