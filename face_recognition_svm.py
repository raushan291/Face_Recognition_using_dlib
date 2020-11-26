# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition
from sklearn import svm
import os
import pickle
import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from config import configuration

config = configuration()

# Training the SVC classifier

def train(training_dir, model_save_path=None):

    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory
    train_dir = os.listdir(training_dir)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(training_dir + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(training_dir + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face, model=config['face_location_model'])

            #If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_loc = face_recognition.face_locations(face, model=config['face_location_model'])
                face_enc = face_recognition.face_encodings(face, face_loc, model=config['face_encoding_model'])[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " was skipped and can't be used for training")
                print('no. of bounding boxes = ', len(face_bounding_boxes))

    test_size=len(encodings)//5  
    X_train, X_test, y_train, y_test = train_test_split(encodings, names, test_size=test_size, random_state=int(time.time() % 100))

    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(X_train, y_train)

    # Save the trained SVM classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(clf, f)

    # predict the response
    start = int(time.time() * 1000000)
    svm_pred = clf.predict(X_test)
    print("SVM: Prediction time = %.2f" % ((int(time.time() * 1000000) - start)/test_size), 'us/prediction')
    print("SVM: Classification Accuracy = %.2f" % (accuracy_score(y_test, svm_pred)))

    return clf


def predict(X_img_path, clf=None, model_path=None):

    if clf is None and model_path is None:
        raise Exception("Must supply svm classifier either thourgh clf or model_path")

    if clf is None:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)

    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(X_img_path)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image, model=config['face_location_model'])
   
    test_image_enc = face_recognition.face_encodings(test_image, face_locations, model=config['face_encoding_model'])
    name = clf.predict(test_image_enc)

    y_proba = clf.predict_proba(test_image_enc)
    prob = []
    for p in y_proba:
        prob.append(max(p))

    return [X_img_path, name, prob]


def train_svm_model():
    print("Training SVM classifier...")
    train(config['train_dir_path'], model_save_path=config['svm_model_path'])
    print("SVM training complete!")

def svm_predictions(rgb_frame=None):
    prediction = predict(rgb_frame, model_path=config['svm_model_path'])
    return prediction
