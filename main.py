import face_recognition_knn as knn
import face_recognition_svm as svm
from find_facial_features_in_picture import find_facial_features
import json
import face_recognition
import os
import numpy as np
from config import configuration
import cv2
import img2pdf

config = configuration()

train = False
predict = True
mode = ['file', 'video', 'camera']
mode = mode[0]

# Get all trained face encodings
def get_trained_face_encodings():
    path = config['train_dir_path']
    trained_dir = os.listdir(path)
    trained_face_encodings = {}
    for trained_name in trained_dir:
        trained_name = path+trained_name
        enc_list = []
        images = os.listdir(trained_name)
        for image in images:
            image = trained_name+'/'+image
            trained_image = face_recognition.load_image_file(image)
            trained_image_face_loc = face_recognition.face_locations(trained_image, model=config['face_location_model'])
            trained_image_face_encoding = face_recognition.face_encodings(trained_image, trained_image_face_loc, model=config['face_encoding_model'])[0]
            enc_list.append(trained_image_face_encoding.tolist()) # convert numpy array to nested list
            trained_face_encodings[image.split('/')[-2]] = enc_list
    with open(config['encoding_json_path'],'w') as f:
            json.dump(trained_face_encodings, f, indent=2)

# read tarined encodings
def read_encodings():
    with open(config['encoding_json_path'],'r') as f:
        trained_face_enc = json.load(f) 
        for each in trained_face_enc:
            temp = []
            for eachdata in trained_face_enc[each]:
                temp.append(np.array(eachdata))
                trained_face_enc[each] = temp
    return trained_face_enc


# get final results
def final_results(knn_pred, svm_pred):
    trained_face_enc = read_encodings()
    
    image = knn_pred['image_path']
    KNN = []
    SVM = []
    final_detection = None
    closest_distance =None
    pred_output= []

    prob = 0
    for j in range(0, len(knn_pred['probabilities'])):

        KNN.append({'prediction' : knn_pred['predictions'][j], 'probability' : round(knn_pred['probabilities'][j], 2)})
        SVM.append({'prediction' : svm_pred['predictions'][j], 'probability' : round(svm_pred['probabilities'][j], 2)})

        if knn_pred['predictions'][j] == svm_pred['predictions'][j]:
            final_detection = knn_pred['predictions'][j]
            prob = (round(knn_pred['probabilities'][j], 2) + round(svm_pred['probabilities'][j], 2)) / 2.0
            im = face_recognition.load_image_file(image)
            im_face_loc = face_recognition.face_locations(im, model=config['face_location_model'])
            im_face_encoding = face_recognition.face_encodings(im, im_face_loc, model=config['face_encoding_model'])[0]
            face_distances = face_recognition.face_distance(trained_face_enc[final_detection], im_face_encoding)
            closest_distance = face_distances[np.argmin(face_distances)]
            prob = round( (prob + (1-closest_distance)) / 2.0, 2)
            if prob > 0.6:
                pred_output.append({
                    'status' : 'AUTHORIZED',
                    'prediction' : final_detection,
                    'closest_distance' : closest_distance,
                    'probability' : prob
                })
            else:
                pred_output.append({
                    'status' : 'UNAUTHORIZED',
                    'prediction' : 'UNKNOWN',
                    'closest_distance' : closest_distance,
                    'probability' : prob
                })
        else:
            knn_detection = knn_pred['predictions'][j]
            svm_detection = svm_pred['predictions'][j]
            prob = (round(knn_pred['probabilities'][j], 2) + round(svm_pred['probabilities'][j], 2)) / 2.0
            im = face_recognition.load_image_file(image)
            im_face_loc = face_recognition.face_locations(im, model=config['face_location_model'])
            im_face_encoding = face_recognition.face_encodings(im, im_face_loc, model=config['face_encoding_model'])[0]
        
            knn_face_distances = face_recognition.face_distance(trained_face_enc[knn_detection], im_face_encoding)
            knn_closest_distance = knn_face_distances[np.argmin(knn_face_distances)]
            prob = round( (prob + (1-knn_closest_distance)) / 2.0, 2)
            knn_prob = round(knn_pred['probabilities'][j], 2)
            
            svm_face_distances = face_recognition.face_distance(trained_face_enc[svm_detection], im_face_encoding)
            svm_closest_distance = svm_face_distances[np.argmin(svm_face_distances)]
            prob = round( (prob + (1-svm_closest_distance)) / 2.0, 2)
            svm_prob = round(svm_pred['probabilities'][j], 2)

            pred_output.append({
                    'status' : 'UNAUTHORIZED',
                    'knn_prediction' : knn_detection,
                    'svm_prediction' : svm_detection,
                    'probability' : prob,
                    'knn_closest_distance' : knn_closest_distance,
                    'svm_closest_distance' : svm_closest_distance,
                    'knn_probability' : knn_prob,
                    'svm_probability' : svm_prob
                })

    print('image : ',image.split('/')[-1])
    print('KNN : ',KNN,)
    print('SVM : ', SVM)
    print(end='\n')
    
    output = {
                    'image' : image,
                    'KNN' : KNN,
                    'SVM' : SVM
                }
    
    final_output = {
        'image' : image,
        'results': pred_output
    }

    if mode == 'video' or mode == 'camera':
        os.remove(image)

    return output, final_output, im     # We have deleted 'image' from disk so, we need to return numpy array of image i.e, 'im' for further uses.


def write_output_and_final_output_jsons(op, final_op):
    with open(config['output_json_folder']+op['image'].split('/')[-1].split('.')[0]+'.json', 'w') as f:
                json.dump(op, f, indent=2)
    with open(config['final_output_json_folder']+op['image'].split('/')[-1].split('.')[0]+'.json', 'w') as f:
        json.dump(final_op, f, indent=2)

# Train knn and svm model
if train:
    knn.train_knn_model()
    svm.train_svm_model()
    get_trained_face_encodings()

# Prediction
if predict:

    output=[]
    final_output=[]

    if mode == 'video' :
        image_scale = config['image_scale']
        # Open the input movie file
        input_movie = cv2.VideoCapture(config['videoFile'])
        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create an output movie file (make sure resolution/frame rate matches input video!)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (1920,1080)) # 640,480 

        frame_number = 0

        while True:
            # Grab a single frame of video
            ret, frame = input_movie.read()
            frame_number += 1

            # Quit when the input video file ends
            if not ret:
                break
            
            if image_scale != 1:
                resized_frame = cv2.resize(frame, (0, 0), fx=image_scale, fy=image_scale)
            else:
                resized_frame = frame

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = resized_frame[:, :, ::-1]

            cv2.imwrite(config['save_image_from_video']+str(frame_number)+".jpg", rgb_frame)
            rgb_frame = config['save_image_from_video']+str(frame_number)+".jpg"
            print('reading '+str(frame_number)+'/'+str(length))

            predict_knn = knn.knn_predictions(rgb_frame)
            predict_svm = svm.svm_predictions(rgb_frame)

            pred=[]
            for name, (top, right, bottom, left) in predict_knn[1]:
                pred.append(name)

            knn_pred= {'image_path' : rgb_frame,
                                'predictions' : pred,
                                'probabilities' : predict_knn[0]
                                }
                            
            svm_pred= {'image_path' : predict_svm[0],
                                    'predictions' : predict_svm[1],
                                    'probabilities' : predict_svm[2]
                                    }

            find_facial_features(rgb_frame)
            
            op, final_op, im_array = final_results(knn_pred, svm_pred)
            output.append(op)
            final_output.append(final_op)

            write_output_and_final_output_jsons(op, final_op)

            # Label the results
            face_locations = face_recognition.face_locations(im_array)
            for (top, right, bottom, left), opt in zip(face_locations, final_op['results']):
                top *= int(1/image_scale)
                right *= int(1/image_scale)
                bottom *= int(1/image_scale)
                left *= int(1/image_scale)

                # Draw a box around the face
                font = cv2.FONT_HERSHEY_DUPLEX
                
                if opt['status'] == 'AUTHORIZED':
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, opt['prediction'], (left + 6, bottom - 6), font, 1, (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, opt['status'], (left + 6, bottom - 6), font, 1, (255, 255, 255), 2)

            cv2.imshow('Video', cv2.resize(frame, (500,300)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            output_movie.write(frame)
        input_movie.release()
        cv2.destroyAllWindows()


    if mode == 'file':
        test_dir=config['test_dir_path']
        for test_image in os.listdir(test_dir):
            test_img = test_dir+test_image

            predict_knn = knn.knn_predictions(test_img)
            predict_svm = svm.svm_predictions(test_img)

            pred=[]
            for name, (top, right, bottom, left) in predict_knn[1]:
                pred.append(name)

            knn_pred= {'image_path' : test_img,
                                    'predictions' : pred,
                                    'probabilities' : predict_knn[0]
                                    }
                                
            svm_pred= {'image_path' : predict_svm[0],
                                    'predictions' : predict_svm[1],
                                    'probabilities' : predict_svm[2]
                                    }
            
            find_facial_features(test_img)
            op, final_op, _ = final_results(knn_pred, svm_pred)
            output.append(op)
            final_output.append(final_op)

            write_output_and_final_output_jsons(op, final_op)

    with open(config['output_json_path'],'w') as f:
        json.dump(output, f, indent=2)

    with open(config['final_output_json_path'],'w') as f:
        json.dump(final_output, f, indent=2)

    face_imgs=sorted(os.listdir(config['save_facial_fartures']))
    imgs = []
    for each in face_imgs:
        imgs.append(config['save_facial_fartures']+each)
    with open(config['save_facial_fartures_pdf'], "wb") as out_file:
        out_file.write(img2pdf.convert(imgs))
