clear all;
cd('./Capsule');
disp('Training a non-convolutional CapsNet.')
Main_MNIST_MLP_Capsules();

disp('Training a non-convolutional CapsNet.')
Main_MNIST_MLP_DynamicRoutingCapsules();

disp('Training a convolutional CapsNet.')
Main_MNIST_Conv_Capsules();

clear all;
