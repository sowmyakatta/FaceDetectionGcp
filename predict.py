from keras.models import load_model
from keras.preprocessing import image
import numpy as np

class MyPredictor:
    def __init__(self, filename):
        self.filename = filename

    def myPredict(self):
        model = load_model("model.h5")

        test_img = image.load_img(self.filename, target_size=(64,64))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)

        result = model.predict(test_img)
        x = np.nonzero(result[0])

        train_data = {'Anusha': 0, 'Geetha': 1, 'Naveena': 2, 'Ramya': 3, 'Sowmya': 4}

        for name, val in train_data.items():
            if val == x[0][0]:
                return{"image" : name }