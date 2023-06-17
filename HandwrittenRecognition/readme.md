### Handwritten Digit Recognition with Python & CNN

Hello friends, ‘Digits’ are a part of our everyday life, be it License plate on our cars or bike, the price of a product, speed limit on a road, or details associated with a bank account. In the case of a text which is unclear, it is easier to guess the digits in comparison to the alphabets

Machine Learning and Deep Learning are reducing human efforts in almost every field. Moreover, a solution achieved using ML and DL can power various applications at the same time, thereby reducing human effort and increasing the flexibility to use the solution. One such solution is a handwritten digit recognition system that can be used in postal mail sorting, bank check processing, form data entry, etc.

## MNIST DATASET
![alt text](https://techvidvan.com/tutorials/wp-content/uploads/sites/2/2020/09/mnist-dataset.jpg)

We are going to use the famous MNIST dataset for training our CNN model. The MNIST dataset was compiled with images of digits from various scanned documents and then normalized in size. Each image is of a dimension, 28×28 i.e total 784 pixel values.

You do not need to download the dataset from any external source as we will import it from keras.datasets

## Layout of the basic idea
1) Firstly, we will train a CNN (Convolutional Neural Network) on MNIST dataset, which contains a total of 70,000 images of handwritten digits from 0-9 formatted as 28×28-pixel monochrome images.

2) For this, we will first split the dataset into train and test data with size 60,000 and 10,000 respectively.

3) Then, we will preprocess the input data by reshaping the image and scaling the pixel values between 0 and 1.

4) After that, we will design the neural network and train the model.

5) After the model is trained, we will save it for future use.

6) Next, we are going to use a webcam as an input to feed an image of a digit to our trained model.

7) Our model will process the image to identify the digit and return a series of 10 numbers corresponding to the ten digits with an activation on the index of the proposed digit.
