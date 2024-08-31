!pip install tensorflow
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from streamlit_option_menu import option_menu
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


train_dir = r'C:\Users\acer\Downloads\Quality_inspection_ai_project\dataset\casting_data\casting_data\test'
test_dir = r'C:\Users\acer\Downloads\Quality_inspection_ai_project\dataset\casting_data\casting_data\train'

train_datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)


test_data = test_datagen.flow_from_directory(test_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)
for image_batch, labels_batch in train_data:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

  # Check class names
class_names = train_data.class_indices
class_names = list(class_names.keys())
print(class_names)

def get_sample_image(generator):
    images, labels = next(generator)
    image = images[0]
    label_index = np.argmax(labels[0])
    label_name = class_names[label_index]

    return image, label_name


def sample_images(generator, nrows=3, ncols=3):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

    for i in range(nrows*ncols):
        image, label_name = get_sample_image(generator)
        row = i // ncols
        col = i % ncols
        ax = axes[row][col]
        ax.imshow(image)
        ax.set_title(label_name)
        ax.axis('off')

    plt.show()
# Model layers
model = Sequential([
    Conv2D(32, (2, 2), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(2 ,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

result = model.evaluate(test_data)
print("Test loss, Test accuracy : ", result)
# import tensorflow as tf
def main():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Home", "ML_Model", "Dataset","Output"])

    if options =="Home":
        st.title("Quality Inspection")
        st.write("In the manufacturing industry, reducing processing errors in the manufacturing process is important for maximizing profits. In order to reduce processing errors, it is necessary to secure a budget for quality assurance, implement manual inspection work, and review the manufacturing process. Particularly, the inspection process is carried out by many companies, but there are problems such as uneven accuracy denpending on inspection workers and increased labor costs.")
        st.write("This dataset provides image data of impellers for submersible pumps.")
        # URL of the image
        image_url_1 = "https://static.turbosquid.com/Preview/2020/06/07__08_34_27/11R131.JPGB3B4468C-B515-4E11-92F7-4CA67966DB2BZoom.jpg"
        image_url_2="https://5.imimg.com/data5/WI/KC/MY-6121640/submersible-pump-impeller-500x500.jpg"
        # Display the image
        col1, col2 = st.columns(2)
        col1.image(image_url_1, caption='Image 1', use_column_width=True)
        col2.image(image_url_2, caption='Image 2', use_column_width=True)


    elif options =="ML_Model":
        st.title("ML Model")
    elif options== "Dataset":
        st.title("Dataset")
        st.write("The image data is labeled with ok(normal) and def(defect/anomaly) in advance. In addition, since it is necessary to illuminate the image in a stable condition when acquiring the image, the data was acquired based on a special lighting setting.")
        image_url_3 = "https://www.kaggleusercontent.com/kf/116004849/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..cqenhxYiSbKxwF-rhMDIrQ.YCIdfaph6ycW1rwkEbMWVnzzsDEZXM-4sNkdki0V01a1iOrksG_QnatHMl1o7Us8e2_K9RfJtwqW_U-IxJ9RhhTt6SQQTvevcaamGDcfiJO9v9Lfs-B_Zs6qY7-am65WrlM47RbUXwlPWPbZjRxBR-jCPcp70Kj-Aso1pQPcilLaTIAYlJmodHEMsN1GDG4P0sCBxujtPsUGnfmUDFGhPH7IXA_aQd2odHOgVItmxoS14gOKL18UJmeuetEGgep03_wD2DpqJzXHi1AKHzAOa3kriDIgUbugLW-xc7lCrf1A_BA-1nzmX-hNPXD2a1Yr34CDLti1jmXCzbbh_xNb5AykV4nF9cuVinrTWjh37v7_OZJNrrr_44ce7TUoWAM2DRLvCeSWO9rEMP7WB4wdFtpxIpdpl9k7CNiWUxAk5K0mmVQgEP4VJADcvg2UUQ8T97ILtHlPZ-16DgM9Layy5byz9Uo7LRN5bCnCZyMxDGzMUROo1y2AsBC4TGWqgwoZxWx7pwcS88SyvJJyOX3fDqvv1jpCzDp-1KexEmypvRcXJ1fSEZxlyz71K4sHCJmRqGYRwIQoaiRhWsUMJZodxw1NxANloxss3ew2E-lNzsEOXrMqSdj3sBwU8N_xxY2nc08Djt-UEVGzXocMgi2vBQ2TYq7bWszi1DVBjMkUO5DXDohMPCwXfL39BUyh5m4L.1LgPgyAIa2jg5z7Mj4Z8XQ/__results___files/__results___9_0.png"
        # Display the image
        st.image(image_url_3, caption='Ok  defective', use_column_width=False)
        st.write("This is sample of the dataset")
        dataframe=sample_images(train_data, nrows=3, ncols=3)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(dataframe)
    elif options == "Output":
        st.title("Output")
        y_true = test_data.classes
        y_pred = model.predict(test_data)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        # Plot the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.write("Output of the Dataset:")
        images, labels = next(test_data)
        indices = np.random.choice(range(len(images)), size=9)
        images = images[indices]
        labels = labels[indices]

        predictions = model.predict(images)
        class_names=list(test_data.class_indices.keys())


        plt.figure(figsize=(12,12))

        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

            image = images[i]

            if image.shape[-1] == 1:
                image = np.squeeze(image)

            plt.imshow(image)

            predicted_label = np.argmax(predictions[i])

            if predicted_label == np.argmax(labels[i]):
                color='blue'
                result_text="Correct"

            else:
                color='red'
                result_text="Incorrect"

            label_text="True: "+ class_names[np.argmax(labels[i])] + ", Pred: " + class_names[predicted_label] + f" ({result_text})"

            plt.xlabel(label_text,color=color)
            st.pyplot(plt)

if __name__ == "__main__":
    main()
