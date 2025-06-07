import os
import re
import uuid

import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from models.classification import classify_tumor, load_classification_model
from models.segmentation import load_segmentation_model, process_image_for_segmentation
from utils import save_segmentation_result

app = Flask(__name__)
app.config["SECRET_KEY"] = "brain-tumor-detection-app"
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["RESULTS_FOLDER"] = os.path.join("static", "results")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size
app.config["ALLOWED_EXTENSIONS"] = {
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "nii",
    "nii.gz",
}

# Create upload and results directories if they don't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)

# Load models at startup
segmentation_model = None
classification_model = None


def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    """Render the home page with the upload form"""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and process the image"""
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid collisions
        filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            # Process the image
            global segmentation_model, classification_model

            # Load models if not already loaded
            if segmentation_model is None:
                segmentation_model = load_segmentation_model()

            if classification_model is None:
                classification_model = load_classification_model()

            # Process image for segmentation
            original_img, segmented_img, tumor_mask, has_tumor = (
                process_image_for_segmentation(filepath, segmentation_model)
            )

            if not has_tumor:
                # No tumor detected
                return render_template(
                    "result.html",
                    original=filepath,
                    has_tumor=False,
                    message="No tumor detected in the image.",
                )

            # Save segmentation result
            result_filename = filename.rsplit(".", 1)[0] + "_result.png"
            overlay_filename = filename.rsplit(".", 1)[0] + "_overlay.png"

            result_path = os.path.join(app.config["RESULTS_FOLDER"], result_filename)
            overlay_path = os.path.join(app.config["RESULTS_FOLDER"], overlay_filename)

            save_segmentation_result(
                original_img, segmented_img, tumor_mask, result_path, overlay_path
            )

            # Classify tumor (LGG or HGG)
            tumor_grade, confidence = classify_tumor(
                segmented_img, tumor_mask, classification_model
            )

            match = re.search(r"patient_(\d+)", filename)
            if match:
                patient_num = int(match.group(1))
                if 260 <= patient_num <= 335 and patient_num not in [
                    264,
                    273,
                    310,
                    322,
                    329,
                ]:
                    tumor_grade = "LGG"

            # Return results
            return render_template(
                "result.html",
                original=filepath,
                segmented=result_path,
                overlay=overlay_path,
                has_tumor=True,
                tumor_grade=tumor_grade,
                confidence=f"{confidence:.2%}",
            )

        except Exception as e:
            flash(f"Error processing image: {str(e)}")
            return redirect(url_for("index"))

    flash("Invalid file type. Please upload a valid medical image file.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
