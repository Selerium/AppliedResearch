import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense

target_height = 64
target_width = 64
frames_per_clip = 30

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(frames_per_clip, target_height, target_width, channels)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()






# The Conv3D layers are used for 3D convolutions.
# The MaxPooling3D layers perform 3D max pooling.
# The Flatten layer flattens the 3D output to be fed into a traditional dense neural network.
# The output layer has as many neurons as there are classes, with softmax activation for multi-class classification.
# We will adapt model architecture according to our datasets and task. Additionally we need to preprocess data, provide appropriate input shape, number of classes (num_classes), etc.





