import os
import numpy as np
import cv2
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMG_SIZE = 32
DATASET = "training_set"    # download this from kaggle (cat vs dog dataset)

X, Y = [], []

def load_images(folder, label, limit=500):
    count = 0
    for file in os.listdir(folder):
        if file.endswith(".jpg") and count < limit:
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img.flatten())
                Y.append(label)
                count += 1
    print(f"Loaded {count} images from {folder}")

load_images(os.path.join(DATASET, "cats"), 0)
load_images(os.path.join(DATASET, "dogs"), 1)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Perceptron(max_iter=1000)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(Y_test, Y_pred))


# testing.........

test_img_path = "test_image.jpg" 

if os.path.exists(test_img_path):
    img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flat = img.flatten().reshape(1, -1)

    prediction = model.predict(img_flat)[0]
    label = "Dog" if prediction == 1 else "Cat"
    print("Prediction for image is :", label)
else:
    print("Image not found!")
