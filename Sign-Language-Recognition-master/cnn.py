# Convolutional Neural Network

# Importing libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

# Initialising the CNN
classifier = Sequential()

# ---------------- FIRST CONVOLUTION BLOCK ----------------
classifier.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    activation='relu',
    input_shape=(64,64,3)
))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))

# ---------------- SECOND CONVOLUTION BLOCK ----------------
classifier.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    activation='relu'
))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))

# ---------------- THIRD CONVOLUTION BLOCK ----------------
classifier.add(Conv2D(
    filters=128,
    kernel_size=(3,3),
    activation='relu'
))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))

# ---------------- FLATTENING ----------------
classifier.add(Flatten())

# ---------------- FULLY CONNECTED LAYERS ----------------
classifier.add(Dense(units=256, activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(units=128, activation='relu'))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# ---------------- COMPILING ----------------
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------- IMAGE PREPROCESSING ----------------
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'test_dataset/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'test_dataset/test_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# ---------------- TRAINING ----------------
classifier.fit(
    training_set,
    epochs=25,
    validation_data=test_set
)