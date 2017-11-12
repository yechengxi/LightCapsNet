clear all;
cd('./Capsule');
disp('Testing training a non-convolutional CapsNet.')
Main_MNIST_MLP_Capsules();


disp('Testing training a convolutional CapsNet.')
Main_MNIST_Conv_Capsules();

clear all;
