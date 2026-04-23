from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

print("Loading model...")
model = load_model("xray_model.h5")

img_path = "dataset/test/NORMAL/IM-0001-0001.jpeg"

img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

result = model.predict(img_array)
score = result[0][0]

if score > 0.5:
    print("Prediction: PNEUMONIA")
    print("Confidence:", round(score * 100, 2), "%")
else:
    print("Prediction: NORMAL")
    print("Confidence:", round((1 - score) * 100, 2), "%")