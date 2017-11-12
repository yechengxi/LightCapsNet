function [ y, dzdw,dzdb,opts ] = conv_layer_1d( I,kernel,bias,stride,pad,dzdy,opts )
%FAST_CONV Summary of this function goes here
%   Detailed explanation goes here
%calculate three ffts and iffts
dzdw=[];  
dzdb=[]; 

flip_kernel=0;

if isfield(opts,'use_nntoolbox')&&opts.use_nntoolbox==1 %
    
    if isfield(opts,'use_corr')&&opts.use_corr==0
       kernel=flip(kernel,1);
       flip_kernel=1;
    end
    
    kernel_sz=size(kernel);
    if length(kernel_sz)==2,kernel_sz(3)=1;end
    if isempty(pad),pad=[0,0,0,0];end
    PADDING_MODE=0;
    
    
   %nntb interface: nnet.internal.cnn.layer.Convolution2D(name, filterSize, numChannels, numFilters, stride, padding);
   
    if ~isfield(opts,'layer')||length(opts.layer)<opts.current_layer||~isfield(opts.layer{opts.current_layer},'conv1d_nntb')
        product_info=ver('nnet');
        opts.nnet_ver=str2double(product_info.Version);
        if opts.nnet_ver<11
            [f,in,b]=size(I); 
            if pad(1)~=pad(2)    
               I = pad_data_1d(I,pad,[]);
               pad2=pad;
               pad=[0,0,0,0];
               PADDING_MODE=1;
            else
                pad(4)=0;
            end
            opts.layer{opts.current_layer}.conv1d_nntb=nnet.internal.cnn.layer.Convolution2D('conv1d_nntb', [kernel_sz(1),1], kernel_sz(2) ,kernel_sz(3), [stride(1),1], pad(1:2:end));
        else
            opts.layer{opts.current_layer}.conv1d_nntb=nnet.internal.cnn.layer.Convolution2D('conv1d_nntb', [kernel_sz(1),1], kernel_sz(2) ,kernel_sz(3), [stride(1),1], 'manual',pad);
        end
        
        if opts.use_gpu
            opts.layer{opts.current_layer}.conv1d_nntb = setupForGPUPrediction(opts.layer{opts.current_layer}.conv1d_nntb);
        else
            opts.layer{opts.current_layer}.conv1d_nntb =setupForHostPrediction(opts.layer{opts.current_layer}.conv1d_nntb);
        end
    end

    conv1d_nntb=opts.layer{opts.current_layer}.conv1d_nntb;
    conv1d_nntb.Weights.Value=permute(kernel,[1,4,2,3]);
    conv1d_nntb.Bias.Value=permute(bias(:),[3,4,1,2]);    

    if isempty(dzdy)
        y=conv1d_nntb.forward(permute(I,[1,4,2,3]));
        y=permute(y,[1,3,4,2]);
    else 
        I=permute(I,[1,4,2,3]);
        dzdy=permute(dzdy,[1,4,2,3]);
        
        if opts.nnet_ver<11
            y = conv1d_nntb.backward( I, [], dzdy, [] );
            gradients = conv1d_nntb.gradients(I, dzdy);
            y=permute(y,[1,3,4,2]);
            if PADDING_MODE==1
               pad=pad2;
               y=y(1+pad(1):pad(1)+f,:,:);
            end
        else
            [y,gradients] = conv1d_nntb.backward( I, [], dzdy, [] );
            y=permute(y,[1,3,4,2]);            
        end
        
        dzdw=gradients{1}./b;
        dzdw=permute(dzdw,[1,3,4,2]);
        dzdb=reshape(gradients{2}./b,size(bias));
        if(flip_kernel)
            dzdw=flip(dzdw,1);
        end
    end
    return;
end


if ~isfield(opts,'use_corr')||opts.use_corr==1
   kernel=flip(kernel,1);%most existing packages use corr instead of conv 
   flip_kernel=1;
end

[f,in,b]=size(I);    
    
if(~isempty(pad))
    original_size=f;
    f=f+pad(1)+pad(2); 
end

[k,in,out]=size(kernel);    
 
if isempty(dzdy)
    %forward mode, compute the 'valid' convolution using fft
    if(~isempty(pad))
       I = pad_data_1d(I,pad,[]);       
    end
    
    tk=zeros(f,in,out,'like',I);
    tk(1:k,:,:)=kernel;      
    kernel=tk;
    
    opts.layer{opts.current_layer}.fI=fft(I); %store result
    opts.layer{opts.current_layer}.fk=fft(kernel); %store result
    
    y=zeros(f,out,b,'like',I);
    
    for o=1:out
        fft_conv=opts.layer{opts.current_layer}.fI.*opts.layer{opts.current_layer}.fk(:,:,o);
        y(:,o,:)=sum(fft_conv,2);
             
    end
    y=real(ifft(y));
    
    y = y(k:end,:,:);
    if ~isempty(bias)
        bias_p=permute(bias(:),[2,1,3]);
        y=y+bias_p;
    end
    
    
    %%%%strided convolution
    if(max(stride)>1)
        y=y(1:stride(1):end,:,:);
    end
    
    if opts.training~=1
        opts.layer{opts.current_layer}.fI=[];
        opts.layer{opts.current_layer}.fk=[];
    end
        
else
    %%back prop: load the precomputed ffts and proceed with the
    %%computation.
   
 
    [d,out,b]=size(dzdy);    
    td=zeros(f,out,b,'like',dzdy);
    td(k:stride(1):k-1+d1*stride(1),:,:)=dzdy;
    
    dzdy=td;
    clear td;
    fdzdy=fft(dzdy);
    
    %%calculate the 'full' correlation   
    y=zeros(f,in,b,'like',dzdy);%y=dzdx
    fk=permute(opts.layer{opts.current_layer}.fk,[1,3,2]);
    
    for i=1:in        
        fft_corr=fdzdy.*conj(fk(:,:,i));
        y(:,i,:)=sum(fft_corr,2);
    end
    y=real(ifft(y));
    
    %next line is a dirty circular shift, according to matlab fft implementation.
    y=circshift(y,k-1,1); 
               
    if(~isempty(pad))
        y=y(1+pad(1):pad(1)+original_size,:,:);
    end
    
   
    dzdw=zeros(k,in,out,'like',I);
   
    
    for o=1:out
        fft_corr=conj(opts.layer{opts.current_layer}.fI).*fdzdy(:,o,:);
        fft_corr=mean(fft_corr,3); %minibatch averaging
        fft_corr=real(ifft(fft_corr));
        dzdw(:,:,o)= fft_corr(1:k,:,:);%
    end    

    if(flip_kernel)
        dzdw=flip(dzdw,1);
    end
    
        
    if ~isempty(bias)
        dzdb=sum(mean(dzdy,3),1);   
        %minibatch averaging + patch summing (note this is how much it changes the final loss)
        dzdb=permute(dzdb,[2,1,3]);
    end
    
    

end




