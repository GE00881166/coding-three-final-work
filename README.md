
# coding-three-final-work


# github
https://github.com/GE00881166/coding-three-final-work.git

# video
https://b23.tv/arDczbA

# Where do I obtain image data
https://www.google.com.hk/search?q=%E5%8A%A8%E7%89%A9&tbm=isch&ved=2ahUKEwjltu_QkZmDAxXTZfUHHQlBA6QQ2-cCegQIABAA&oq=%E5%8A%A8%E7%89%A9&gs_lcp=CgNpbWcQAzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6CAgAEIAEELEDOg4IABCABBCKBRCxAxCDAToECAAQAzoECAAQHjoGCAAQCBAeUPELWOYpYMgsaAFwAHgBgAGMAogB7g-SAQYxLjEyLjKYAQCgAQGqAQtnd3Mtd2l6LWltZ7ABAMABAQ&sclient=img&ei=Tk-AZeXXD9PL1e8PiYKNoAo&bih=640&biw=1355
Choosing images from Google for training data has two key benefits:

 #1. Google offers a wide range of images covering various categories, enhancing the model's ability to recognize different patterns and features.2. Google provides a large number of images, contributing to a richer and more extensive dataset, which generally leads to better model performance.

# Main references
https://blog.csdn.net/weixin_40863591/article/details/111711591?spm=1001.2014.3001.5501

# references of the model:
https://www.kaggle.com/code/aryaadithyan/convolutional-neural-network-implementation

# Here's some of the documentation I consulted when making changes to the reference programme

https://scikit-learn.org/1.0/modules/generated/sklearn.metrics.plot_confusion_matrix.html
https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
https://www.codesofinterest.com/2017/03/graph-model-training-history-keras.html

# Visualising the history of training for reference
（I couldn't find the URL where I download this reference code, so I will upload it to GitHub and indicate which part of the code I referenced.）
#ref: https://pan.baidu.com/s/1wNGHETG9_Jp7cpukqEHh6Q?pwd=2xgk PIN: 2xgk

# Loss calculation for multiple classifications:
https://keras.io/api/losses/
https://keras.io/api/losses/probabilistic_losses/#sparse_categorical_crossentropy-function



# Main references of image_manager (Batch move files python code implementation)
https://blog.csdn.net/weixin_40863591/article/details/129048705?spm=1001.2014.3001.5501





# Design and development processes
For the `Train_model` program, the first module is the import section. I referred to the import section of the "Cat-Dog Image Classification" code and imported the machine learning libraries I might need. The first one is Numpy, a Python library for mathematical and scientific computing. It assists in numerical operations. The second is Pandas, which simplifies data visualization. The third is OpenCV, used for image processing, and the fourth is the Pyplot module from Matplotlib, providing MATLAB-style functions for plotting charts. Additionally, I imported os to perform operations on files and folders in the Python environment. Following the code reference, I chose to use the Keras framework and skipped importing zipfile since I didn't have compression or decompression needs.

Moving on to the second module, creating training data, I referred to a tutorial (https://blog.csdn.net/weixin_40863591/article/details/111711591?spm=1001.2014.3001.5501). I started by defining the folder "data" for reading data. After obtaining categories based on the names of image directories, I initialized variables X and Y. The process involves iterating through all images, determining their original categories, reading grayscale images using the imread module from OpenCV, and resizing images to (80, 80). These variables are then filled into X and Y, where X represents the matrix data of each image, and Y represents the category of each image. The next steps involve looping through each image, following the reference's procedures, and normalizing data by compressing pixel values to the range of 0-1 and reshaping the matrix. I noticed that the model built based on this reference was not very accurate.

The third step involves building the model, where I referred to the "Cat-Dog Image Classification" approach for the basic model construction. However, since I am dealing with multi-class classification, I consulted other documents to adapt the loss calculation for binary classification to handle more label classes. I also switched the activation function from sigmoid to softmax.

In the fourth step, visualizing training history, I referenced a program for "Daily Fashion Image Classification." I incorporated its data visualization methods into my program, starting with presenting the training history as a line plot and visualizing the confusion matrix. After this, we can proceed to train the data.After completing the training, we can observe that, after 100 epochs, the accuracy is approximately around 65%.

For the `image_manager` program, as we have already completed the model training, the main part to reference is the implementation of batch file movement. I began by importing the necessary machine learning libraries, following the "Implementing Batch Move" example, which involved importing the shutil library. The subsequent steps include loading the pre-trained model and label_encoder. Referring to the "Implementing Batch Move" example, I constructed image file paths, similar to the previous `train_model` approach, to read and preprocess images. The images are then passed to the model for prediction. To determine the most probable category for each image, I used the argmax method from the numpy library to get the prediction's maximum value. The label_encoder is then employed to obtain the name of the most likely category for each image, and these results are added to the `results` variable. Now, I am ready to create folders, following the guidance from "Implementing Batch Classification." After running the program, we can observe that the model effectively classifies images into different folders.