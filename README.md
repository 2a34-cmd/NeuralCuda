NeuralCuda 
===============

the project involves 
1-artificial neural network (NN) trainer written in Cuda (for Nvidia GPU side) and C++ (for CPU side)
2-web server to use NN for digit recognition written in C
3-mn1 file format for encoding the weights and biases of NN
4-a Python script for generating random NN with specific layers

The prime goals of the project are open-sourcing NN trainer with frontend (which is chosen to be a website) while using the high-level language that gives complete control and management (which is C/C++/Cuda) and using lowest possible number of external libraries without using it in no meaningful way and for example using python for scriptting and html/css/js for website frontend.
================
TRAINER
the trainer uses Nvidia GPU to train (change weights and biases to minimize error of) NN. The prime goal is to use C language while minimizing dependencies without loss of convenience. However, the NVCC compiler is a modified C++ compiler so the whole trainer was painted with C++ even for the CPU side. In addition to training, the trainer minimizes the learning rate to raise the NN efficiency in an automatic way to fine-tune the NN.
================
Python script and (mn1/mn2) format
mni (multineural version i) format is text that has a line for every possible construction and connection inside NN. The parser inside both the (c++ parser needs version 1 or 2)trainer and (c parser needs version 2) webserver use different versions for the format and forward compatibility. The whole difference between mn1 and mn2 includes the number of layers inside the neural network line and the number of neurons inside the layer line. The idea here is to minimize memory footprint and allocation/deallocation steps (space complexity). The Python script is exactly what it sounds; a way to generate prespecified-shape NN with random weights and biases. It eases the process of initiating NN for training purposes.
===============
WEB SERVER
unlike the trainer, the web server is written completely in C. It uses std libraries (stdio,stdlib,...) and pthread and math libraries. As it sounds, the math library was needed for evaluating the activation function and pthread was needed for creating a thread pool for handling clients in the queue (async programming). The server uses the producer (enqueuer)- consumer (handlers) model.
==============

Don't expect maintaining or adding features in timed manner. This project was and will be a personal product from me for the world to see.
