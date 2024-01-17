# MindColor 🧠🌊
MindColor is a simple GAN that takes brain waves as input, and attempts to recreate the image that the person was looking at while its mind was being recorded.
This is my final project for class "Machine Learning Advanced Models".

## Dataset

### Data
Brainwaves
- brainwaves were collected using a meditation headband, made by Flowtime [link](https://www.meetflowtime.com/);
- 5 different types of waves: alpha, betha, gamma, delta, theta

Images
- images were generated with python and matplotlib;
- four categories of images: red, blue, green, yellow as squares.
  
In detail I have recorded my brain activity while staring at each image, for about 15 minutes. 
Data is collected by the headband every 0.6 seconds.

### Dataset
Two dataset were created:
1. one for the generator, made of brainwaves
2. one for the discriminator, made of images

## Model

A simple GAN was implemented. The generator input is an array ...
