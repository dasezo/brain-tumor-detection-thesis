import os
import pickle

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

# Path to the SVM model
SVM_MODEL = "/home/dasezo/dev/project/models/svm_model.pkl"


def load_classification_model():
    """Load the SVM classification model"""
    try:
        with open(SVM_MODEL, "rb") as file:
            model_data = pickle.load(file)
        print("Loaded SVM model from", SVM_MODEL)
        return model_data
    except Exception as e:
        print(f"Error loading classification model: {str(e)}")
        raise


def extract_features(segmented_img, tumor_mask):
    """
    Extract features from the segmented tumor image for classification

    These features are similar to those used in training the SVM model.
    """
    features = {}

    # Create masks for different tumor components
    # In the segmentation output, classes are:
    # 0: Background, 1: Necrotic/Core, 2: Edema, 3: Enhancing

    # Extract tumor components from segmentation
    segmented_classes = np.zeros(
        (segmented_img.shape[0], segmented_img.shape[1]), dtype=np.uint8
    )

    # Extract class from BGR color
    for y in range(segmented_img.shape[0]):
        for x in range(segmented_img.shape[1]):
            pixel = segmented_img[y, x]
            if np.array_equal(pixel, [0, 0, 255]):  # Red (BGR)
                segmented_classes[y, x] = 1  # Necrotic/Core
            elif np.array_equal(pixel, [0, 255, 0]):  # Green (BGR)
                segmented_classes[y, x] = 2  # Edema
            elif np.array_equal(pixel, [255, 0, 0]):  # Blue (BGR)
                segmented_classes[y, x] = 3  # Enhancing

    # Create masks for each class
    ncr_net_mask = segmented_classes == 1
    ed_mask = segmented_classes == 2
    et_mask = segmented_classes == 3
    tc_mask = ncr_net_mask | et_mask
    wt_mask = tumor_mask  # Whole tumor is the entire tumor mask

    # Volume features
    voxel_volume = 1.0  # Voxel volume in mm³ (assuming isotropic 1mm voxels)
    features["ncr_net_volume"] = np.sum(ncr_net_mask) * voxel_volume
    features["ed_volume"] = np.sum(ed_mask) * voxel_volume
    features["et_volume"] = np.sum(et_mask) * voxel_volume
    features["tc_volume"] = np.sum(tc_mask) * voxel_volume
    features["wt_volume"] = np.sum(wt_mask) * voxel_volume

    # Volume ratios
    if features["wt_volume"] > 0:
        features["tc_wt_ratio"] = features["tc_volume"] / features["wt_volume"]
        features["et_tc_ratio"] = (
            features["et_volume"] / features["tc_volume"]
            if features["tc_volume"] > 0
            else 0
        )
        features["ncr_net_tc_ratio"] = (
            features["ncr_net_volume"] / features["tc_volume"]
            if features["tc_volume"] > 0
            else 0
        )
        features["ed_wt_ratio"] = features["ed_volume"] / features["wt_volume"]
    else:
        features["tc_wt_ratio"] = 0
        features["et_tc_ratio"] = 0
        features["ncr_net_tc_ratio"] = 0
        features["ed_wt_ratio"] = 0

    # Shape features (approximations based on 2D image)
    if np.sum(wt_mask) > 0:
        # Project along axes
        y_projection = np.sum(wt_mask, axis=1)
        x_projection = np.sum(wt_mask, axis=0)

        # Extent along each axis
        features["wt_y_extent"] = np.sum(y_projection > 0)
        features["wt_x_extent"] = np.sum(x_projection > 0)

        # Calculate elongation and flatness
        major_axis = max(features["wt_y_extent"], features["wt_x_extent"])
        minor_axis = min(features["wt_y_extent"], features["wt_x_extent"])

        features["wt_elongation"] = major_axis / minor_axis if minor_axis > 0 else 0
        features["wt_flatness"] = 0  # Cannot calculate in 2D
        features["wt_sphericity"] = 0  # Cannot calculate in 2D
    else:
        features["wt_y_extent"] = 0
        features["wt_x_extent"] = 0
        features["wt_elongation"] = 0
        features["wt_flatness"] = 0
        features["wt_sphericity"] = 0

    # Intensity features (using grayscale values from segmented image)
    gray_segmented = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

    # Calculate intensity statistics for whole tumor
    if np.sum(wt_mask) > 0:
        wt_intensities = gray_segmented[wt_mask]
        features["wt_intensity_mean"] = np.mean(wt_intensities)
        features["wt_intensity_std"] = np.std(wt_intensities)
        features["wt_intensity_max"] = np.max(wt_intensities)
        features["wt_intensity_min"] = np.min(wt_intensities)
    else:
        features["wt_intensity_mean"] = 0
        features["wt_intensity_std"] = 0
        features["wt_intensity_max"] = 0
        features["wt_intensity_min"] = 0

    # Add more features similar to those in the SVM training code
    # Here we're using a simplified set based on what we can extract from a single 2D image

    return features


def classify_tumor(segmented_img, tumor_mask, model_data):
    # Extract features from the segmented image
    features = extract_features(segmented_img, tumor_mask)

    # Get all expected feature names from the scaler
    all_features = model_data["scaler"].feature_names_in_

    # Create a complete feature vector with zeros for any missing features
    features_array = np.zeros((1, len(all_features)))

    # Fill in the features we have
    for i, feature_name in enumerate(all_features):
        if feature_name in features:
            features_array[0, i] = features[feature_name]

    # Apply scaling
    X_scaled = model_data["scaler"].transform(features_array)

    # Select only the top features for prediction
    top_feature_indices = [
        list(all_features).index(feature) for feature in model_data["top_features"]
    ]
    X_selected = X_scaled[:, top_feature_indices]

    # Make prediction
    svm_model = model_data["svm_model"]
    prediction = svm_model.predict(X_selected)[0]
    probability = svm_model.predict_proba(X_selected)[0][1]  # Probability of being HGG

    tumor_grade = "HGG" if prediction == 1 else "LGG"
    return tumor_grade, probability
