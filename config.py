def configuration():

    config = {
        'videoFile' : '/home/rakumar/FR/test.mp4',
        'image_scale': 0.5,

        'face_location_model' : 'hog',    # 'cnn'    or  'hog'
        'face_encoding_model' : 'small',  # 'large'  or  'small'

        'train_dir_path' : '/home/rakumar/FR/train_dir/',
        'test_dir_path' : '/home/rakumar/FR/test_dir/',

        'encoding_json_path' : '/home/rakumar/FR/encoding.json',

        'output_json_folder': '/home/rakumar/FR/output_json/',
        'final_output_json_folder': '/home/rakumar/FR/final_output_json/',

        'output_json_path' : '/home/rakumar/FR/output.json',
        'final_output_json_path' : '/home/rakumar/FR/final_output.json',

        'knn_model_path' : '/home/rakumar/FR/trained_knn_model.clf',
        'svm_model_path' : '/home/rakumar/FR/trained_svm_model.clf',

        'save_image_from_video' : '/home/rakumar/FR/save_image_from_video/',
        'save_facial_fartures' : '/home/rakumar/FR/save_facial_features/',
        'save_facial_fartures_pdf' : '/home/rakumar/FR/out.pdf'
    }

    return config