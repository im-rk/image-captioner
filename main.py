import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid pre-allocating all memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


with tf.device('/CPU:0'):
    vgg_model = VGG16(weights="imagenet")
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

model = tf.keras.models.load_model('mymodel.h5')

with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

st.set_page_config(page_title="Caption Generator App", page_icon="📷")

st.title("Image Caption Generator")
st.markdown("Upload an image, and this app will generate a caption for it using a trained LSTM model.")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    st.subheader("Generated Caption")
    with st.spinner("Generating caption..."):
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        image_features = vgg_model.predict(image, verbose=0)

        max_caption_length = 34
        
        def get_word_from_index(index, tokenizer):
            return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                if predicted_word is None:
                    break
                caption += " " + predicted_word
                if predicted_word == "endseq":
                    break
            return caption

        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")

    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{generated_caption.strip()}”</p>'
        f'</div>',
        unsafe_allow_html=True
    )