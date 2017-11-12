function [ opts ] = PrepareData_MNIST_CNN( opts )
%PREPAREDATA_MNIST_MLP Summary of this function goes here
%   Detailed explanation goes here


imdb=get_mnist(opts);

opts.train=imdb.images.data(:,:,:,imdb.images.set==1);
opts.train_labels=imdb.images.labels(imdb.images.set==1);
opts.test=imdb.images.data(:,:,:,imdb.images.set==3);
opts.test_labels=imdb.images.labels(imdb.images.set==3);
opts.n_class=max(opts.train_labels(:));
opts.n_train=size(opts.train,4);
opts.n_test=size(opts.test,4);

end

