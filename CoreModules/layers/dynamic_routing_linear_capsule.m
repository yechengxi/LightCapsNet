function [y,dzdw,B,opts] = dynamic_routing_linear_capsule(x,weight,B,d1,d2,dzdy,opts)
%DYNAMIC_ROUTING_LINEAR_CAPSULE Summary of this function goes here
%   Detailed explanation goes here
dzdw=[];  
layer_idx=opts.current_layer;

if isempty(dzdy)
    %ff
    n1=size(x,1)/d1;%input capsules
    n2=size(weight,1)/d2;%output capsules
    
    %softmax
    E = exp(B- max(B,[],2)) ;
    S = sum(E,2) ;
    C = E./S;
    opts.layer{layer_idx}.C=C;
    
    C=repelem(C,d2,d1);    
    y=(C.*weight)*x;    
    %eq_check1=weight*x;
    
    opts.layer{layer_idx}.x1=y;

    %squash
    y = squash(y,d2,[]);

    %calculate U
    weight=reshape(weight,size(weight,1),d1,[]);%reshaped as d2n2 x d1 x n1
    x=reshape(x,1,d1,n1,[]);                    %reshaped as    1 x d1 x n1 x batch 
    U=sum(weight.*x,2);                         %            d2n2 x 1  x n1 x batch
    U=permute(U,[1,3,4,2]);                     %reshaped as d2n2 x n1 x batch
    %eq_check2=permute(sum(U,2),[1,3,2]);
    U=reshape(U,d2,n2,n1,[]);                   %reshaped as d2   x n2 x n1 x batch
    
    S=reshape(y,d2,n2,1,[]);                    %reshaped as d2   x n2 x 1  x batch
    opts.layer{layer_idx}.A=permute(mean(sum(S.*U,1),4),[2,3,1]);%calculate A
    
else
    %bp
    [dzdy] = squash(opts.layer{layer_idx}.x1,d2,dzdy);
    C=repelem(opts.layer{layer_idx}.C,d2,d1);
    y=(C.*weight)'*dzdy;    
    dzdy=permute(dzdy,[1,3,2]);
    x=permute(x,[3,1,2]);
    dzdw=dzdy.*x;
    dzdw=mean(dzdw,3);
    B=B+opts.layer{layer_idx}.A;%update
    
end

end

