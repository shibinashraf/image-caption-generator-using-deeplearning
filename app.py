import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.utils import load_img
from keras.preprocessing import image, sequence
import cv2
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from gtts import gTTS
import os
from pathlib import Path


vocab = np.load('./vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


print("+"*50)
print("vocabulary loaded")

embedding_size = 128
vocab_size = len(vocab)
max_len = 40


image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))


conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('./mine_model_weights.h5')

print("="*150)
print("MODEL LOADED")

#resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
resnet = load_model('./resnet.h5')

print("="*150)
print("RESNET MODEL LOADED")


st.header("Image Caption Generator📸")
image_file = st.file_uploader("Upload Images",type=["png","jpg","jpeg"])


save_folder = './static'
save_path = Path(save_folder,"file.jpg")
if image_file is not None:
	with open(save_path, mode='wb') as w:
	    w.write(image_file.getvalue())
if image_file is not None:
    # TO See details
    file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
	#st.write(file_details)
    st.image(load_img(image_file), width=250)
    if st.button('predict'):
        image = cv2.imread('static/file.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224))
        image = np.reshape(image, (1,224,224,3))
        incept = resnet.predict(image).reshape(1,2048)
        print("="*50)
        print("Predict Features")

        text_in = ['startofseq']

        final = ''

        print("="*50)
        print("GETING Captions")

        count = 0
        while tqdm(count < 20):

            count += 1

            encoded = []
            for i in text_in:
                encoded.append(vocab[i])

            padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

            sampled_index = np.argmax(model.predict([incept, padded]))

            sampled_word = inv_vocab[sampled_index]

            if sampled_word != 'endofseq' and sampled_word != '.' :
                final = final + ' ' + sampled_word

            text_in.append(sampled_word)
        st.subheader(final)
        language = 'en'
        myobj = gTTS(text=final, lang=language, slow=False)
        myobj.save("static/caption.mp3")
        audio_file = open('.\static\caption.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')

st.subheader("or else you can use the any of the images below. Just drag and drop them.")
st.image(load_img("surf.jpg"), width=250)
st.image(load_img("download.jpg"), width=250)
st.image(load_img("images.jpg"), width=250)
