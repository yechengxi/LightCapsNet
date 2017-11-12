clear all;
addpath(genpath('../CoreModules'));
n_epoch=20; %training epochs
dataset_name='mnist'; %dataset name
network_name='mlp'; %network name
use_gpu=(gpuDeviceCount>0) %use gpu or not 
if use_gpu
    %Requires Neural Network Toolbox to use it.
    opts.use_nntoolbox=license('test','neural_network_toolbox')
end
%function handle to prepare your data
PrepareDataFunc=@PrepareData_MNIST_CNN;
%function handle to initialize the network
NetInit=@net_init_mnist_conv_capsules;

%automatically select learning rates
use_selective_sgd=0; 
%select a new learning rate every n epochs
ssgd_search_freq=10; 
learning_method=@adam; %training method: @sgd,@rmsprop,@adagrad,@adam
%opts.parameters.mom=0.9;
opts.parameters.clip=1e1;
%sgd parameter 
%(unnecessary if selective-sgd is used)
sgd_lr=1e-3;

opts.parameters.weightDecay=0;%1e-3;
opts.parameters.batch_size=500;
opts.n_class=10;
Main_Template(); %call training template