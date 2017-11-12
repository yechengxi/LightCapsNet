function net = net_init_mnist_conv_capsules(opts)
rng('default');
rng(0) ;
input=28;%mnist
k1=9;
k2=9;

hidden1=256;

caps_n1=32;
caps_d1=8;

caps_n2=10;
caps_d2=16;

f=1/100;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv2d', ...
                           'weights', {{f*randn(k1,k1,1,hidden1, 'single'), zeros(hidden1,1,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu');
input=input-k1+1;

%simulate a linear capsule layer
net.layers{end+1} = struct('type', 'conv2d', ...
                           'weights', {{f*randn(k2,k2,hidden1,caps_n1*caps_d1, 'single'), zeros(caps_n1*caps_d1,1,'single')}}, ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'squash','d',caps_d1);% squash capsules 
input=(input-k1+1)/2;

%simulate a linear capsule layer
net.layers{end+1} = struct('type', 'conv2d', ...
                           'weights', {{f*randn(input,input,caps_n1*caps_d1,caps_n2*caps_d2, 'single'), zeros(caps_n2*caps_d2,1,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'squash','d',caps_d2);

net.layers{end+1} = struct('type', 'marginloss','m',0.1,'M',0.9,'lambda',0.5,'d',caps_d2) ;
