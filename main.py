import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define the labels for the classes
labels = ['dandelion', 'daisy', 'tulip', 'sunflower', 'rose']
# Define the size of the images
img_size = 128

# Initialize the data and labels lists
x = []
y = []

# Load the images and labels
for label in labels:
    data = os.path.join("./input/", label)
    for image in os.listdir(data):
        try:
            # Read the image and resize it
            im = cv2.imread(os.path.join(data, image), cv2.IMREAD_COLOR)
            im = cv2.resize(im, (img_size, img_size))

            # Append the image and label to the data and labels lists
            x.append(im)
            y.append(labels.index(label))
        except Exception as e:
            pass

# Convert the data and labels lists to numpy arrays
x = np.array(x)
y = np.array(y)

# Normalize the data (scale the images to a range between 0 and 1)
x = x / 255.0
# Reshape the data to the shape required by the CNN
x = x.reshape(-1, img_size, img_size, 3)

# Convert the labels to one-hot encoding
y = to_categorical(LabelEncoder().fit_transform(y), 5)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

# Add the layers to the model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))
# Define the data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
)
datagen.fit(x_train)

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"])
model.summary()

# Define the callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.1)

# Train the model
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                              epochs=30, validation_data=(x_test, y_test),
                              callbacks=[early_stopping, learning_rate_reduction],
                              steps_per_epoch=x_train.shape[0] // 32,
                              verbose=1)

# Predict the labels of the test set
y_pred = model.predict(x_test)

# Evaluate the model
print("Test Accuracy: {0:.2f}%".format(model.evaluate(x_test, y_test)[1] * 100))

# Save the model
model.save('sequential_model.h5')

# Plot the training history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val'])
plt.show()
