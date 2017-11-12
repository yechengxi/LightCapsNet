function [y] = capsule_margin(x,c,M,m,lambda,d,dzdy)
%CAPSULE_MARGIN Summary of this function goes here
%   Detailed explanation goes here

batch_dim=length(size(x));
if batch_dim>2 %2d cnn
    x=squeeze(x);
end

n_class=size(x,1)/d;
batch_size=size(x,2);
L=reshape(x,d,[]);
L=sum(L.^2,1).^0.5;    
L=reshape(L,n_class,[]);
idx=c(:)'+n_class*[0:batch_size-1];%ground truth idx
mask=zeros(n_class,batch_size,'logical');
mask(idx)=true;
    
if isempty(dzdy)
    y=(sum(max(M-L(mask),0).^2)+lambda*(sum(max(L(~mask)-m,0).^2)))/batch_size;
    
else
    mask=repelem(mask,d,1);
    L=repelem(L+eps,d,1);
    mask1=(mask)&(L<M);
    mask2=(~mask)&(L>m);
    x=x./L;
    y=2*(L-M).*x.*mask1+2*lambda*x.*(L-m).*mask2;
    if batch_dim>2 %2d cnn
       y=permute(y,[3,4,1,2]);
    end

    
end
end

