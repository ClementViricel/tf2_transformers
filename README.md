# Transformer architecture in Keras tensorflow 2.0
## Introduction
This project is about using new Tensorflow 2.0 and Keras to construct your own neural machine translation with transformer architecture (Scientific paper [here](https://arxiv.org/abs/1706.03762 "Attention is all you need").  
Tensorflow 2.0 is the brend new version of Tensorflow that is more user-friendly than the previous version. It also incorporate a Keras high Level API. 
In this project, there is two way of training your own neural machine translation:  
1. Using tensorflow tf.data API  
2. Using the old fashion way with lists of inputs/outputs  

### With tensorflow data API
The tensorflow data API allow you to process data (image, text etc.) properly with a simple API.  
It also allow to use generator in order to save RAM (very usefull when you are dealing with a lot of data).  
You can find the documentation [here](https://www.tensorflow.org/guide/data "Build TensorFlow input pipelines").  

### With a list of inputs
WIP... 
