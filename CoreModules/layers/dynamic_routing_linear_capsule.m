function [y,dzdw,opts] = dynamic_routing_linear_capsule(x,weight,d1,d2,dzdy,opts)
%DYNAMIC_ROUTING_LINEAR_CAPSULE Summary of this function goes here
%   Detailed explanation goes here
dzdw=[];  
layer_idx=opts.current_layer;

if ~isfield(opts,'rout_it')
    opts.rout_it=2;
end

if isempty(dzdy)
    
    
    %ff
    [n1,batch_size]=size(x);
    n1=n1/d1;%input capsules
    n2=size(weight,1)/d2;%output capsules
    B=zeros(n2,n1,batch_size,'like',x);
    
    for r=1:opts.rout_it
        %softmax
        E = exp(B- max(B,[],2)) ;
        S = sum(E,2) ;
        C = E./S;
        
        %calculate U
        weight1=reshape(weight,size(weight,1),d1,[]);%reshaped as d2n2 x d1 x n1
        x1=reshape(x,1,d1,n1,[]);                    %reshaped as    1 x d1 x n1 x batch 
        U=sum(weight1.*x1,2);                        %            d2n2 x 1  x n1 x batch
        U=permute(U,[1,3,4,2]);                      %reshaped as d2n2 x n1 x batch
        
        %%%*dynamic routing*%%%
        S=squeeze(sum(repelem(C,d2,1).*U,2));  
        y = squash(S,d2,[]);%v in the paper
        
        V=reshape(y,d2,n2,1,[]);                     %reshaped as d2   x n2 x 1  x batch
        U=reshape(U,d2,n2,n1,[]);                    %reshaped as d2   x n2 x n1 x batch
        A=permute(sum(V.*U,1),[2,3,4,1]);
        
        B=B+A;
    end
    
    opts.layer{layer_idx}.S=S;
    opts.layer{layer_idx}.C=C;

else
    %bp
    [dzdy] = squash(opts.layer{layer_idx}.S,d2,dzdy);
    C=repelem(opts.layer{layer_idx}.C,d2,d1);
    dzdy=permute(dzdy,[1,3,2]);
    y=squeeze(sum((weight.*C).*dzdy,1));    
    x=permute(x,[3,1,2]);
    dzdw=dzdy.*x.*C;
    dzdw=mean(dzdw,3);
    
end

end

