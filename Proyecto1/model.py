# Libraries used
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifiergit 
from sklearn.metrics import accuracy_score
from joblib import dump

print("Extraxting letters...")
# Extracting all the photos to train the model
dir = '<YOUR_DIRECTION_TO_THE_FOLDER_WITH_ALL_THE_CHARACTERS>'
searcher = os.listdir(dir)
X = []
y = []

for i in range(len(searcher)):
    photos = os.listdir(f'{dir}/{searcher[i]}')
    for j in range(len(photos)):
        letter = cv.imread(f'{dir}/{searcher[i]}/{photos[j]}', cv.IMREAD_GRAYSCALE)
        letter_reduced = letter.flatten()
        X.append(letter_reduced)
        y.append(searcher[i])

print("Splitting datasets...")
# Train - test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100, stratify=y)

print("Training the model...")
# Using random forest at first
model = RandomForestClassifier(n_estimators = 100, max_features = 45)
model.fit(X_train, y_train)

print("Executing model...")
# Predicting values and checking accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Making persistent the model
dump(model, "model.sav")
print("Process finished!")