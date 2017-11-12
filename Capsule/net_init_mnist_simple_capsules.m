function net = net_init_mnist_simple_capsules(opts)
rng('default');
rng(0) ;
input=28*28;
hidden1=256;
caps_n1=32;
caps_d1=8;

caps_n2=10;
caps_d2=16;

f=1/100;
net.layers = {} ;
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{f*randn(hidden1,input, 'single'), zeros(hidden1,1,'single')}}) ;
net.layers{end+1} = struct('type', 'relu');
%
net.layers{end+1} = struct('type', 'linear_capsule', 'd',caps_d1,'n',caps_n1, ...
                           'weights', {{f*randn(caps_n1*caps_d1,hidden1, 'single'), zeros(caps_n1*caps_d1,1,'single')}}) ;
net.layers{end+1} = struct('type', 'squash','d',caps_d1); 
net.layers{end+1} = struct('type', 'linear_capsule', 'd',caps_d2,'n',caps_n2, ...
                           'weights', {{f*randn(caps_n2*caps_d2,caps_n1*caps_d1, 'single'), zeros(caps_n2*caps_d2,1,'single')}}) ;
net.layers{end+1} = struct('type', 'squash','d',caps_d2);

net.layers{end+1} = struct('type', 'marginloss','m',0.1,'M',0.9,'lambda',0.5,'d',caps_d2) ;
