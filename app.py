import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

import os

model = load_model('model_covid.h5')

def main():
	"""Face Detection App"""

	st.title("Covid19 Detection From XRay Images")
	st.text("Build with Streamlit and DeepLearning")
	image_file = st.file_uploader("Upload XRay Image",type=['jpg','png','jpeg'])
	if image_file is not None:
		img_data = Image.open(image_file)
		img = img_data.resize((224, 224), Image.BILINEAR) 
		st.image(img)
		#img=image.load_img(image(image_data),target_size=(224,224))
		img=image.img_to_array(img)
		img=np.expand_dims(img,axis=0)
		pred=model.predict_classes(img)
		if pred[0][0] == 0:
			st.error("COVID 19 is Detected")
		else:
			st.info("Congratulation! No Covid19 is Detected")


if __name__ == '__main__':
		main()	




