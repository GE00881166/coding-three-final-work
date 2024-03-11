import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


import joblib


from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D


#import warnings


#warnings.filterwarnings("ignore", category=UserWarning)

#ref:https://blog.csdn.net/weixin_40863591/article/details/111711591?spm=1001.2014.3001.5501
main_dir = "data/"



def get_category(root):
    #category = root.split(os.path.sep)[1]
    category = root.split('/')[1]
    return category


def create_test_data(path):

    X = []

    y = []

    #ref:https://www.runoob.com/python/os-walk.html
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'DS' in file:
                continue
            #filepath = os.path.join(root, file)
            filepath = root + "/" + file

            category = get_category(filepath)

            img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            new_img_array = cv2.resize(img_array, dsize=(80, 80))
            #
            X.append(new_img_array)
            y.append(category)
    return X, y



def process_data(X, y):

    X = np.array(X).reshape(-1, 80, 80, 1)
    y = np.array(y)
    X = X / 255.0
    return X, y


#ref:https://www.kaggle.com/code/aryaadithyan/convolutional-neural-network-implementation
def build_model(class_num,X_shape):


    model = Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add another:
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # Add a softmax layer
    model.add(Dense(class_num, activation='softmax'))
    # ref:https://keras.io/api/losses/
    # ref:https://keras.io/api/losses/probabilistic_losses/#sparse_categorical_crossentropy-function
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


#ref: https://pan.baidu.com/s/1wNGHETG9_Jp7cpukqEHh6Q?pwd=2xgk PIN: 2xgk
def plot_confusion_matrix(y_true, y_pred, class_names):
    plt.figure(figsize=(15, 8))

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
   # plt.show()




X, y = create_test_data(main_dir)

X, y_labels = process_data(X, y)

print(X.shape)

class_num = len(np.unique(y))


label_encoder = LabelEncoder()


y  = label_encoder.fit_transform(y_labels)


model = build_model(class_num ,X.shape[1:])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=20, validation_split=0.2)


evaluation_result = model.evaluate(X_test,y_test)




print(f"Loss: {evaluation_result[0]:.4f}")
print(f"Accuracy: {evaluation_result[1]*100:.2f}%")


#ref : https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# https://www.codesofinterest.com/2017/03/graph-model-training-history-keras.html

plt.figure(figsize=(15,8))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1)
plt.savefig("keras_learning_curves_plot.png")




y_pred = np.argmax(model.predict(X_test), axis=1)


decoded_y_test = label_encoder.inverse_transform(y_test)
decoded_y_pred = label_encoder.inverse_transform(y_pred)



class_names = label_encoder.classes_
print(class_names)
plot_confusion_matrix(decoded_y_test, decoded_y_pred, class_names)


print("Classification Report:")
print(classification_report(decoded_y_test, decoded_y_pred))


model.save("model.h5")
print("Model saved successfully!")



joblib.dump(label_encoder, "label_encoder.joblib")
print("LabelEncoder saved successfully!")




