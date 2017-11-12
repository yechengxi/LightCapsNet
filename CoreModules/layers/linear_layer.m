function [ y, dzdw,dzdb,opts] = linear_layer( I,weight,bias,dzdy,opts)
%FAST_MLP_LAYER Summary of this function goes here
%   Detailed explanation goes here
%I: input_dim x batch_size

dzdw=[];  
dzdb=[];  

if exist('opts','var')&&isfield(opts,'use_nntoolbox')&&opts.use_nntoolbox==1     
    [out,in]=size(weight);
    batch_size=size(I,2);
    %use nntoolbox
    if ~isfield(opts,'layer')||length(opts.layer)<opts.current_layer||~isfield(opts.layer{opts.current_layer},'fc_nntb')
        product_info=ver('nnet');
        opts.nnet_ver=str2double(product_info.Version);
        if opts.nnet_ver<11
            opts.layer{opts.current_layer}.fc_nntb=nnet.internal.cnn.layer.Convolution2D('conv2d_nntb', [1,1], in ,out, [1,1], [0,0]);
        else
            opts.layer{opts.current_layer}.fc_nntb=nnet.internal.cnn.layer.Convolution2D('conv2d_nntb', [1,1], in ,out, [1,1],'manual', [0,0,0,0]);
        end
        if opts.use_gpu
            opts.layer{opts.current_layer}.fc_nntb = setupForGPUPrediction(opts.layer{opts.current_layer}.fc_nntb);
        else
            opts.layer{opts.current_layer}.fc_nntb =setupForHostPrediction(opts.layer{opts.current_layer}.fc_nntb);
        end
    end
    fc_nntb=opts.layer{opts.current_layer}.fc_nntb;
    if ~isempty(bias) 
        fc_nntb.Bias.Value=permute(bias(:),[3,4,1,2]);
    else
        fc_nntb.Bias.Value=zeros(1,1,out,1);
    end
    fc_nntb.Weights.Value=permute(weight,[4,3,2,1]);
    I=permute(I,[3,4,1,2]);

    if isempty(dzdy)  
        y=fc_nntb.forward(I);
        y=permute(y,[3,4,1,2]);
    else
        
        dzdy=permute(dzdy,[3,4,1,2]);
        if opts.nnet_ver<11
            y = fc_nntb.backward(  I, [], dzdy, [] );
            gradients = fc_nntb.gradients(I, dzdy);
        else
            [y,gradients] = fc_nntb.backward( I, [], dzdy, [] );
        end
        
        dzdw=gradients{1}./batch_size;
        dzdb=gradients{2}./batch_size;
        y=permute(y,[3,4,1,2]);
        dzdw=permute(dzdw,[4,3,2,1]);
        dzdb=reshape(dzdb,size(bias));
    end
    
    return;
end


if isempty(dzdy)
    %forward mode

    y=weight*I;
    
    if ~isempty(bias)
        y=y+bias;        
    end
    
else    
    %backward mode
    
    y=weight'*dzdy;    
    if ~isempty(bias)
        dzdb=mean(dzdy,2);%minibatch averaging    
    end    
    dzdy=permute(dzdy,[1,3,2]);
    I=permute(I,[3,1,2]);
    dzdw=dzdy.*I;
    dzdw=mean(dzdw,3);
    
end




