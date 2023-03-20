from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
interpreter = None
input_index = None
output_index = None

class_names = ['CBB', 'CBSD', 'CGM', 'CMD', 'HEALTHY']

BUCKET_NAME = "cassava-model-3" # Here you need to put the name of your GCP bucket


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "model3/cassava_output.h5",
            "/tmp/cassava_output.h5",
        )
        model = tf.keras.models.load_model("/tmp/cassava_output.h5")

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((128, 128)) # image resizing
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 4)

    return {"class": predicted_class, "confidence": confidence}