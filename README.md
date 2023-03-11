# image-caption-generator-using-deeplearning

## Table of Contents

- [Description](#description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Description
This project is a deep learning-based system that generates captions for images and converts them into audio using the ResNet50 and LSTM models. The ResNet50 model is used for image feature extraction, and the LSTM model is used for natural language processing. The system takes an image as input, extracts its features using ResNet50, generates a textual caption using LSTM, and then converts the caption into audio using text-to-speech (TTS) technology. The end result is an audio description of the image that can be useful for visually impaired individuals or for enhancing the user experience of multimedia content.

## Dataset 
https://www.kaggle.com/datasets/adityajn105/flickr8k

Consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. … The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations.

## Installation

- Set up a virtual environment: Create a virtual environment to keep your project's dependencies separate from your system's dependencies. You can use tools like virtualenv or conda to create a virtual environment.

```
  pip install virtualenv
  virtualenv <my_env_name>
  source <my_env_name>/bin/activate
```


- Install dependencies: Once you have set up your virtual environment, install the required dependencies for your project using a package manager like pip or conda. You can either install them all at once by using a requirements.txt file or manually install them one by one.

```
   pip install -r requirements.txt
```

- Download pre-trained models: Download pre-trained models, you may need to download them separately and add them to your project's directory.

## Usage
 Run the project using following command
 ```
   streamlit run app.py
```
Upload any image and click submit. It'll automatically generate text and audio caption for the image.

!! Will not work properly with images other than the ones which are provided in the dataset.
