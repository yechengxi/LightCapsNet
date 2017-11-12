function [ y,from,opts ] = maxpool_1d( I, K, S,pad,dzdy,from,opts )

if isfield(opts,'use_nntoolbox')&&opts.use_nntoolbox==1 %
    if isempty(pad),pad=[0,0,0,0];end
    
    I=permute(I,[1,4,2,3]);pad(4)=0;
    PADDING_MODE=0;
    
    if ~isfield(opts,'layer')||length(opts.layer)<opts.current_layer||~isfield(opts.layer{opts.current_layer},'maxpool_nntb')
        K=[K,1];S=[S,1];
        product_info=ver('nnet');
        opts.nnet_ver=str2double(product_info.Version);
        if opts.nnet_ver<11.0
            if pad(1)~=pad(2)||pad(3)~=pad(4)    
               [i1,i2,in,b]=size(I);  
               I = pad_data(I,pad,[]);
               pad2=pad;
               pad=[0,0,0,0];
               PADDING_MODE=1;
            end
            opts.layer{opts.current_layer}.maxpool_nntb = nnet.internal.cnn.layer.MaxPooling2D( 'maxpool_nntb', K,S,pad(1:2:end));

        else
            opts.layer{opts.current_layer}.maxpool_nntb = nnet.internal.cnn.layer.MaxPooling2D( 'maxpool_nntb', K,S,'manual',pad);
        end
        
        if opts.use_gpu
            opts.layer{opts.current_layer}.maxpool_nntb = setupForGPUPrediction(opts.layer{opts.current_layer}.maxpool_nntb);
        else
            opts.layer{opts.current_layer}.maxpool_nntb =setupForHostPrediction(opts.layer{opts.current_layer}.maxpool_nntb);
        end
    end
   
    
    maxpool_nntb=opts.layer{opts.current_layer}.maxpool_nntb ;
    
    if isempty(dzdy)
        y=maxpool_nntb.forward(I);
        from=y;
    else
        dzdy=permute(dzdy,[1,4,2,3]);
        y= maxpool_nntb.backward( I, from, dzdy, []);
        if PADDING_MODE==1
           pad=pad2;
           y=y(1+pad(1):pad(1)+i1,1+pad(3):pad(3)+i2,:,:);
        end
    end
    
    y=permute(y,[1,3,4,2]);
        
    return;
end




if exist('opts','var')
   if ~isfield(opts,'parameters')||~isfield(opts.parameters,'eps_pool')
      opts.parameters.eps_pool=0.0; 
   end
else
    opts.training=0;
    opts.parameters.eps_pool=0.0;
end

if isempty(dzdy)
    %%forward
    
    if(~isempty(pad))
       I = pad_data_1d(I,pad,[]);       
    end
    
    [Din,N,B]=size(I);
    
    Dout = ceil((Din-K+1)/S);
    
    [slices,idx0]=unroll_ln(I,K,S);
    [y,from]=max(slices,[],1);
    from=double(from);%
    y=reshape(y,Dout,N,B);
    from=reshape(from,Dout,N,B);
            
    %now we need to deal with the indexes again.
    idx0=reshape(idx0(1,:,:,:)-1,size(from));
    from=from+idx0;
    
    
else
    %backward
    %dzdy=gather(dzdy);
    
    input_size=size(I);
    Din=input_size(1);
    
    if(~isempty(pad))
        input_size(1)=input_size(1)+pad(1)+pad(2); 
    end

    y=zeros(input_size,'like',dzdy);
    %%%
    
    if(K<=S)
        y(from)=dzdy;%this is faster but can be wrong.
    else
        y=accumarray(from(:),dzdy(:),[prod(input_size),1]);    %fast and correct
        y=reshape(y,input_size);
    end
    
         
    if(~isempty(pad))
        y=y(1+pad(1):pad(1)+Din,:,:);
    end
    
end