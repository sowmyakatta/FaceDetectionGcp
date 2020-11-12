#CNN classifier creation
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D

classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation="relu"))
classifier.add(MaxPool2D((2,2)))
classifier.add(Conv2D(64, (3,3), activation="relu"))
classifier.add(MaxPool2D((2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=120, activation="relu"))
classifier.add(Dense(units= 80, activation="relu"))
classifier.add(Dense(units=5, activation="softmax"))
classifier.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])

#Input data shaping
from keras.preprocessing.image import ImageDataGenerator
train_gen =ImageDataGenerator(rescale=1./255, shear_range= 0.2,
                              horizontal_flip= True, vertical_flip= True, zoom_range= 0.2)

test_gen = ImageDataGenerator(rescale=1./255)

train_data =train_gen.flow_from_directory(directory=r"C:\Users\HELLO\Desktop\DS\my_dog_cat\Persons",
                                          target_size=(64,64), color_mode="rgb", class_mode="categorical")

test_data =train_gen.flow_from_directory(directory=r"C:\Users\HELLO\Desktop\DS\my_dog_cat\Persons",
                                          target_size=(64,64), color_mode="rgb", class_mode="categorical")


#fitting the data into classifier and saving it to disk

model = classifier.fit_generator(train_data,steps_per_epoch= 50, epochs=1, validation_data= test_data,
                                 validation_steps=20, verbose= 1)

classifier.save("model.h5")
print("Model save to disk")


#prediction
from keras.preprocessing import image
import numpy as np

test_img = image.load_img(r"C:\Users\HELLO\Desktop\test1.jpg", target_size=(64,64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
result = classifier.predict(test_img)
x = np.nonzero(result[0])

for name, val in train_data.class_indices.items():
    if val == x[0][0]:
        print(name)
