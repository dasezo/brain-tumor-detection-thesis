import os

import cv2
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Path to the segmentation model
UNET_WEIGHTS = "/home/dasezo/dev/project/models/best_model.weights.h5"
UNET_MODEL = "/home/dasezo/dev/project/models/seg_model.keras"

# Segmentation classes
SEGMENT_CLASSES = {
    0: "NOT tumor",
    1: "NECROTIC/CORE",  # or NON-ENHANCING tumor CORE
    2: "EDEMA",
    3: "ENHANCING",  # original 4 -> converted into 3
}

# Size that the model expects
IMG_SIZE = 128


def load_segmentation_model():
    """Load and compile the U-Net segmentation model"""
    try:
        # Try to load the full model first
        if os.path.exists(UNET_MODEL):
            model = load_model(UNET_MODEL, compile=False)
            print("Loaded full U-Net model from", UNET_MODEL)
        else:
            # Build model architecture (same as in the training code)
            from tensorflow.keras.layers import (
                Conv2D,
                Dropout,
                Input,
                MaxPooling2D,
                UpSampling2D,
                concatenate,
            )
            from tensorflow.keras.models import Model

            inputs = Input((IMG_SIZE, IMG_SIZE, 2))
            ker_init = "he_normal"
            dropout = 0.2

            # Encoder
            conv1 = Conv2D(
                32, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(inputs)
            conv1 = Conv2D(
                32, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(conv1)
            pool = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv = Conv2D(
                64, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(pool)
            conv = Conv2D(
                64, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(conv)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv)

            conv2 = Conv2D(
                128, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(pool1)
            conv2 = Conv2D(
                128, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(
                256, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(pool2)
            conv3 = Conv2D(
                256, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(conv3)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv5 = Conv2D(
                512, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(pool4)
            conv5 = Conv2D(
                512, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(conv5)
            drop5 = Dropout(dropout)(conv5)

            # Decoder
            up7 = Conv2D(
                256, 2, activation="relu", padding="same", kernel_initializer=ker_init
            )(UpSampling2D(size=(2, 2))(drop5))
            merge7 = concatenate([conv3, up7], axis=3)
            conv7 = Conv2D(
                256, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(merge7)
            conv7 = Conv2D(
                256, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(conv7)

            up8 = Conv2D(
                128, 2, activation="relu", padding="same", kernel_initializer=ker_init
            )(UpSampling2D(size=(2, 2))(conv7))
            merge8 = concatenate([conv2, up8], axis=3)
            conv8 = Conv2D(
                128, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(merge8)
            conv8 = Conv2D(
                128, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(conv8)

            up9 = Conv2D(
                64, 2, activation="relu", padding="same", kernel_initializer=ker_init
            )(UpSampling2D(size=(2, 2))(conv8))
            merge9 = concatenate([conv, up9], axis=3)
            conv9 = Conv2D(
                64, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(merge9)
            conv9 = Conv2D(
                64, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(conv9)

            up = Conv2D(
                32, 2, activation="relu", padding="same", kernel_initializer=ker_init
            )(UpSampling2D(size=(2, 2))(conv9))
            merge = concatenate([conv1, up], axis=3)
            conv = Conv2D(
                32, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(merge)
            conv = Conv2D(
                32, 3, activation="relu", padding="same", kernel_initializer=ker_init
            )(conv)

            conv10 = Conv2D(4, (1, 1), activation="softmax")(conv)

            model = Model(inputs=inputs, outputs=conv10)

            # Load weights
            model.load_weights(UNET_WEIGHTS)
            print("Built U-Net model and loaded weights from", UNET_WEIGHTS)

        # Compile the model (matching the training configuration)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )

        return model

    except Exception as e:
        print(f"Error loading segmentation model: {str(e)}")
        raise


def preprocess_image(image_path):
    """Preprocess an image for the segmentation model"""
    try:
        # Check if the file is a NIfTI file
        if image_path.endswith(".nii") or image_path.endswith(".nii.gz"):
            # Load NIfTI file
            nifti_img = nib.load(image_path)
            volume = nifti_img.get_fdata()

            # Extract middle slice from the volume
            middle_idx = volume.shape[2] // 2
            flair_slice = volume[:, :, middle_idx]
            t1ce_slice = volume[
                :, :, middle_idx
            ]  # Ideally would be a different modality

            # Resize to model input size
            flair_slice = cv2.resize(flair_slice, (IMG_SIZE, IMG_SIZE))
            t1ce_slice = cv2.resize(t1ce_slice, (IMG_SIZE, IMG_SIZE))

        else:
            # For regular images (png, jpg, etc.), load and duplicate channel
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Convert to grayscale if it's a color image
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img

            # Resize to model input size
            img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))

            # Since our model expects two channels (FLAIR and T1CE),
            # we'll use the same image for both channels as a simplification
            flair_slice = img_resized
            t1ce_slice = img_resized

        # Normalize images
        flair_norm = (
            flair_slice / np.max(flair_slice)
            if np.max(flair_slice) > 0
            else flair_slice
        )
        t1ce_norm = (
            t1ce_slice / np.max(t1ce_slice) if np.max(t1ce_slice) > 0 else t1ce_slice
        )

        # Stack channels for model input
        X = np.zeros((1, IMG_SIZE, IMG_SIZE, 2))
        X[0, :, :, 0] = flair_norm
        X[0, :, :, 1] = t1ce_norm

        # Store original image for visualization
        original_image = img_gray if "img_gray" in locals() else flair_slice

        return X, original_image

    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise


def process_image_for_segmentation(image_path, model):
    """
    Process an image with the segmentation model

    Returns:
    - original_img: The original input image
    - segmented_img: Image with segmentation (classes colored)
    - tumor_mask: Binary mask where tumor regions are 1, otherwise 0
    - has_tumor: Boolean indicating if a tumor was detected
    """
    # Preprocess the image
    X, original_img = preprocess_image(image_path)

    # Run prediction
    prediction = model.predict(X)[0]

    # Get class with highest probability for each pixel
    segmented = np.argmax(prediction, axis=-1)

    # Create colored segmentation image
    segmented_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Define colors for each class (BGR format)
    colors = {
        0: [0, 0, 0],  # Black for background
        1: [0, 0, 255],  # Red for necrotic/core
        2: [0, 255, 0],  # Green for edema
        3: [255, 0, 0],  # Blue for enhancing
    }

    # Create tumor mask (any non-zero class)
    tumor_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    tumor_mask[segmented > 0] = 1

    # Check if tumor was detected (at least some percentage of pixels)
    tumor_percentage = np.sum(tumor_mask) / (IMG_SIZE * IMG_SIZE)
    has_tumor = tumor_percentage > 0.01  # At least 1% of pixels should be tumor

    # Apply colors to segmentation
    for class_id, color in colors.items():
        segmented_img[segmented == class_id] = color

    # Resize original image to match segmentation for visualization
    if original_img.shape != (IMG_SIZE, IMG_SIZE):
        original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

    # Ensure original image is 8-bit for display
    if original_img.dtype != np.uint8:
        original_img = (original_img * 255).astype(np.uint8)

    return original_img, segmented_img, tumor_mask, has_tumor
