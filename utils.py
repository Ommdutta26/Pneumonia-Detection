import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# load SavedModel
model = tf.saved_model.load("model2")
infer = model.signatures["serving_default"]

IMG_SIZE = 224


def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    return img_array


def predict_image(image):

    img_array = preprocess_image(image)

    prediction = infer(tf.constant(img_array))
    prediction = list(prediction.values())[0].numpy()

    prob = float(prediction[0][0])

    if prob > 0.5:
        return "PNEUMONIA", prob
    else:
        return "NORMAL", prob


# -------- GradCAM -------- #

def make_gradcam_heatmap(img_array):

    with tf.GradientTape() as tape:

        inputs = tf.constant(img_array)
        tape.watch(inputs)

        preds = infer(inputs)
        preds = list(preds.values())[0]

    grads = tape.gradient(preds, inputs)

    heatmap = tf.reduce_mean(grads, axis=-1)[0]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


def apply_gradcam(image):

    img_array = preprocess_image(image)

    heatmap = make_gradcam_heatmap(img_array)

    # resize heatmap
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)

    # convert to color map (3 channels)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # original image
    original = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

    # ensure same type
    original = original.astype(np.uint8)

    # overlay
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return superimposed