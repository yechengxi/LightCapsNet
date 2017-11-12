function [ y,opts] = lrn( x,N,kappa,alpha,beta,dzdy,opts)
%LRN Summary of this function goes here
%   Detailed explanation goes here

    if isfield(opts,'use_nntoolbox')&&opts.use_nntoolbox==1 %
        if ~isfield(opts,'layer')||length(opts.layer)<opts.current_layer||~isfield(opts.layer{opts.current_layer},'lrn_nntb')
            opts.layer{opts.current_layer}.lrn_nntb = nnet.internal.cnn.layer.LocalMapNorm2D( 'lrn_nntb', N,alpha,beta,kappa);
            if opts.use_gpu
                opts.layer{opts.current_layer}.lrn_nntb = setupForGPUPrediction(opts.layer{opts.current_layer}.lrn_nntb);                
            else
                opts.layer{opts.current_layer}.lrn_nntb =setupForHostPrediction(opts.layer{opts.current_layer}.lrn_nntb);
            end
        end   
        lrn_nntb=opts.layer{opts.current_layer}.lrn_nntb ;

        if isempty(dzdy)
            y=lrn_nntb.forward(x);
        else
            y=lrn_nntb.forward(x);
            y=lrn_nntb.backward(x,y,dzdy,[]);
        end    

       return; 
    end
    
    sz=size(x);
    if (length(sz)>2)
        channel_dim=3; Index={':',':'};%cnn
    else
        channel_dim=1; Index={};%mlp
    end

    y=zeros(size(x),'like',x);
    L=y;
    x2_cumu=cumsum(x.^2,channel_dim);
    

    n_channels=size(x,channel_dim);
    for n=1:n_channels
       nStart=max(n-floor((N-1)/2),1)-1;       
       nEnd=min(n+ceil((N-1)/2),n_channels);
       Index_new=[Index,n,':'];
       Index_End=[Index,nEnd,':'];
       Index_Start=[Index,nStart,':']; 
       cumu_start=0;
       if nStart~=0 
           cumu_start=x2_cumu(Index_Start{:});
       end
       L(Index_new{:})=kappa+alpha*(x2_cumu(Index_End{:}) -cumu_start);            
       
    end

    
    if isempty(dzdy)
       y=x./L.^(beta);
    else
       
        temp_cumu=cumsum(dzdy.*x./L.^(1+beta),channel_dim);
        LL=zeros(size(L),'like',x);
        for n=1:size(x,channel_dim)
            nStart=max(n-floor((N-1)/2),1)-1;       
            nEnd=min(n+ceil((N-1)/2),n_channels);
            Index_new=[Index,n,':'];
            Index_End=[Index,nEnd,':'];
            Index_Start=[Index,nStart,':'];  
            cumu_start=0;
           if nStart~=0 
               cumu_start=temp_cumu(Index_Start{:});
           end
           LL(Index_new{:})=x(Index_new{:}).*(temp_cumu(Index_End{:})-cumu_start);
        end
    
       y=dzdy./L.^(beta)-2*alpha*beta.*LL;
   end

    
end

