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

A simple GAN was implemented. The generator input is an array of floats, length is 5 (one datapoint for each kind of brainwave). Its output is a rgb image 28x28x3. The discriminator input is the image generated and its output is a "judgement", which is a float that can be positive or negative. Loss function is cross_entropy for either the generator and the discriminator.

## Results
Training Loss

![Loss_v2](https://github.com/enaikey00/MindColor/assets/64537810/6f16c08c-cef9-4661-abb5-a949f2ee974c)

Generated Images

![image_at_epoch_0001](https://github.com/enaikey00/MindColor/assets/64537810/39d8db30-a847-4c6b-9b06-44a19f563fad)

Test Loss

![Loss_Eval_10Epochs](https://github.com/enaikey00/MindColor/assets/64537810/1aad40af-ff92-4c98-bde6-aff24a4d5a4b)

Generated Images

![image_at_epoch_0001](https://github.com/enaikey00/MindColor/assets/64537810/cf4feee6-4965-4834-812e-9a3d6730a8ad)

## Improvements
- bigger input matrix for generator (ex. 50x5 instead of 1x5); this means more data has to be collected
- we could predict the color of the image instead of generating the image; this shift the task to a classifcation one, and ligthens the model (since we don't need 28x28x3 but just a few pixels)
- another model
- more epochs (like 1000s)
