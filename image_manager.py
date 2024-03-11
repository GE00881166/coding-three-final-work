


import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import joblib
import shutil


loaded_model = load_model("model.h5")


loaded_label_encoder = joblib.load("label_encoder.joblib")


test_folder = "images"


output_folder = "output_folder"



results = []


for root, dirs, files in os.walk(test_folder):
    for file in files:
        if 'DS' in file:  # 忽略系统文件
            continue


        file_path = os.path.join(root, file)


        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        processed_img = np.array(new_img_array).reshape(-1, 80, 80, 1) / 255.0


        predicted_class = np.argmax(loaded_model.predict(processed_img), axis=1)
        predicted_class_label = loaded_label_encoder.inverse_transform(predicted_class)


        results.append({
            'File Path': file_path,
            'Predicted Class': predicted_class_label[0]
        })


        target_folder = os.path.join(output_folder, predicted_class_label[0])


        os.makedirs(target_folder, exist_ok=True)


        target_file_path = os.path.join(target_folder, file)


        shutil.move(file_path, target_file_path)


#results_df = pd.DataFrame(results)


#results_df.to_csv("classification_results.csv", index=False)
#print("Classification results saved to 'classification_results.csv'")
