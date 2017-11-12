function [ y, dzdw,dzdb,opts ] = conv_layer_2d( I,kernel,bias,stride,pad,dzdy,opts )
%FAST_CONV Summary of this function goes here
%   Detailed explanation goes here
%calculate three ffts and iffts
dzdw=[];  
dzdb=[]; 

flip_kernel=0;


if isfield(opts,'use_nntoolbox')&&opts.use_nntoolbox==1 %
    
    if isfield(opts,'use_corr')&&opts.use_corr==0
       kernel=flip(flip(kernel,1),2);
       flip_kernel=1;
    end
    
    [k1,k2,k3,k4]=size(kernel);
    if isempty(pad),pad=[0,0,0,0];end
    
    
   %nntb interface: nnet.internal.cnn.layer.Convolution2D(name, filterSize, numChannels, numFilters, stride, padding);
    if ~isfield(opts,'layer')||length(opts.layer)<opts.current_layer||~isfield(opts.layer{opts.current_layer},'conv2d_nntb')
        product_info=ver('nnet');
        opts.nnet_ver=str2double(product_info.Version);
        if opts.nnet_ver<11
            PADDING_MODE=0;
            if pad(1)~=pad(2)||pad(3)~=pad(4)    
               [i1,i2,in,b]=size(I);  
               I = pad_data(I,pad,[]);
               pad2=pad;
               pad=[0,0,0,0];
               PADDING_MODE=1;
            end
            opts.layer{opts.current_layer}.conv2d_nntb=nnet.internal.cnn.layer.Convolution2D('conv2d_nntb', [k1,k2], k3,k4, [stride(1),stride(2)], pad(1:2:end));
        else
            opts.layer{opts.current_layer}.conv2d_nntb=nnet.internal.cnn.layer.Convolution2D('conv2d_nntb', [k1,k2], k3,k4, [stride(1),stride(2)], 'manual',pad);
        end
        if opts.use_gpu
            opts.layer{opts.current_layer}.conv2d_nntb = setupForGPUPrediction(opts.layer{opts.current_layer}.conv2d_nntb);
        else
            opts.layer{opts.current_layer}.conv2d_nntb =setupForHostPrediction(opts.layer{opts.current_layer}.conv2d_nntb);
        end
    end

    conv2d_nntb=opts.layer{opts.current_layer}.conv2d_nntb;
    conv2d_nntb.Weights.Value=kernel;
    conv2d_nntb.Bias.Value=permute(bias(:),[3,4,1,2]);    

    if isempty(dzdy)
        
        y=conv2d_nntb.forward(I);    
    else         
        if opts.nnet_ver<11
            y = conv2d_nntb.backward( I, [], dzdy, [] );
            gradients = conv2d_nntb.gradients(I, dzdy);
            if PADDING_MODE==1
               pad=pad2;
               y=y(1+pad(1):pad(1)+i1,1+pad(3):pad(3)+i2,:,:);
            end
        else
            [y,gradients] = conv2d_nntb.backward( I, [], dzdy, [] );
        end
        
        dzdw=gradients{1}./opts.parameters.batch_size;
        dzdb=reshape(gradients{2}./opts.parameters.batch_size,size(bias));
        if(flip_kernel)
            dzdw=flip(flip(dzdw,1),2);
        end
    end
    return;
end

if ~isfield(opts,'use_corr')||opts.use_corr==1
   kernel=flip(flip(kernel,1),2);%most existing packages use corr instead of conv 
   flip_kernel=1;
end

[i1,i2,in,b]=size(I);    
    
if(~isempty(pad))
    original_size_r=i1;
    original_size_c=i2;
    i1=i1+pad(1)+pad(2);
    i2=i2+pad(3)+pad(4);
end

[k1,k2,in,out]=size(kernel);    
 
if isempty(dzdy)
    %forward mode, compute the 'valid' convolution using fft
    if(~isempty(pad))
       I = pad_data(I,pad,[]);       
    end
    
    tk=zeros(i1,i2,in,out,'like',I);
    tk(1:k1,1:k2,:,:)=kernel;      
    kernel=tk;
    
    opts.layer{opts.current_layer}.fI=fft2(I); %store result
    opts.layer{opts.current_layer}.fk=fft2(kernel); %store result
    
   
    y=zeros(i1,i2,out,b,'like',I);
    
    for o=1:out
        fft_conv=opts.layer{opts.current_layer}.fI.*opts.layer{opts.current_layer}.fk(:,:,:,o);
        y(:,:,o,:)=sum(fft_conv,3);
    end
    y=real(ifft2(y));     

    y = y(k1:end,k2:end,:,:);
    if ~isempty(bias)
        bias_p=permute(bias(:),[4,3,1,2]);% check this
        y=y+bias_p;
    end
    
    
    %%%%strided convolution
    if(max(stride)>1)
        y=y(1:stride(1):end,1:stride(2):end,:,:);
    end
    
    
    if opts.training~=1
        opts.layer{opts.current_layer}.fI=[];
        opts.layer{opts.current_layer}.fk=[];
    end
        
else
    %%back prop: load the precomputed ffts and proceed with the
    %%computation.
   
    %%calculate the 'valid' correlation+flipping    
 
    [d1,d2,out,b]=size(dzdy);
    
    td=zeros(i1,i2,out,b,'like',dzdy);
    
    td(k1:stride(1):k1-1+d1*stride(1),k2:stride(2):k2-1+d2*stride(2),:,:)=dzdy;
    dzdy=td;
    clear td;
    fdzdy=fft2(dzdy);
    
    
    %%calculate the 'full' correlation   
    y=zeros(i1,i2,in,b,'like',dzdy);%y=dzdx
    fk=permute(opts.layer{opts.current_layer}.fk,[1,2,4,3]);
    
    for i=1:in        
        fft_corr=conj(fk(:,:,:,i)).*fdzdy;
        y(:,:,i,:)=sum(fft_corr,3);
    end
    y=real(ifft2(y));
    
               
    if(~isempty(pad))
        y=y(1+pad(1):pad(1)+original_size_r,1+pad(3):pad(3)+original_size_c,:,:);
    end
    
    
    %%%dzdw
    dzdw=zeros(k1,k2,in,out,'like',I);
    
    
    for o=1:out
        fft_corr=conj(opts.layer{opts.current_layer}.fI).*fdzdy(:,:,o,:);
        fft_corr=mean(fft_corr,4); %minibatch averaging
        fft_corr=real(ifft2(fft_corr));
        dzdw(:,:,:,o)= fft_corr(1:k1,1:k2,:,:);%
    end    

    if(flip_kernel)
        dzdw=flip(flip(dzdw,1),2);
    end

    
    if ~isempty(bias)
        dzdb=sum(sum(mean(dzdy,4),1),2);   
        %minibatch averaging + patch summing (note this is how much it changes the final loss)
        dzdb=permute(dzdb,[4,3,2,1]);
    end
    
    

end




