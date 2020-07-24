import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
from PIL import Image
import warnings
import logging
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image


def predict(image_path, model, top_k):
    img = Image.open(image_path)
    img = np.asarray(img)
    img = process_image(img)
    img = np.expand_dims(img, axis=0)
    ps = model.predict(img)
    ps = ps[0]
    classes = np.argpartition(ps,-top_k)[-top_k:] + 1
    probs = ps[np.argpartition(ps,-top_k)[-top_k:]]
    
    
    return probs, classes

parser = argparse.ArgumentParser(
    description = "predict flowers using your model and a picture to predict"
)

parser.add_argument('Image_path', help='Image path')
parser.add_argument('saved_model', help='provide a model')
parser.add_argument('--top_k', help='top predictions', type=int, default=3)
parser.add_argument('-cn', '--category_names', help='Path to a JSON file mapping labels to flower names', default='label_map.json')

args = parser.parse_args()
image_path = args.Image_path
model = args.saved_model
top_k = args.top_k
json_file = args.category_names
with open(json_file, 'r') as f:
    class_names = json.load(f)



reloaded_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
probs, classes = predict(image_path, reloaded_model, top_k)
labels = []
for i in classes:
    labels.append(class_names[str(i)])
print("|----------------------------------|")
temp_prob = 0
temp_class = ""
for i in range(len(probs)):
    if probs[i] > temp_prob:
        temp_prob = probs[i]*100
        temp_class = labels[i]
        
    print(labels[i]+": "+str(round((probs[i]*100), 3))+"%") 
    
print("|----------------------------------|")

print("The predicted image is %s with %.3f%% confidence" % (temp_class, temp_prob))
    

