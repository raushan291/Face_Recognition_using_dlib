from PIL import Image, ImageDraw
import face_recognition
import os
from config import configuration

config = configuration()

def find_facial_features(img):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(img)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    # Create a PIL imagedraw object so we can draw on the picture
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], fill=(0, 255, 0) ,width=2)

    # Show the picture
    # pil_image.show()

    save_img = config['save_facial_fartures']+img.split('/')[-1]
    pil_image.save(save_img)
