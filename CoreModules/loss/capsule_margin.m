function [y] = capsule_margin(x,c,M,m,lambda,d,dzdy)
%CAPSULE_MARGIN Summary of this function goes here
%   Detailed explanation goes here
    n_class=size(x,1)/d;
    batch_size=size(x,2);
    L=reshape(x,d,[]);
    L=sum(L.^2,1).^0.5;    
    L=reshape(L,n_class,[]);
    idx=c(:)'+n_class*[0:batch_size-1];%ground truth idx
    mask=zeros(n_class,batch_size,'logical');
    mask(idx)=true;
    
if isempty(dzdy)
    %y=(sum(max(M-L(mask),0).^2))/batch_size;
    %y=mean(max(M-L(mask),0).^2);
    y=(sum(max(M-L(mask),0).^2)+lambda*(sum(max(L(~mask)-m,0).^2)))/batch_size;
    
else
    mask=repelem(mask,d,1);
    L=repelem(L+eps,d,1);
    mask1=(mask)&(L<M);
    mask2=(~mask)&(L>m);
    x=x./L;
    y=2*(L-M).*x.*mask1+2*lambda*x.*(L-m).*mask2;
    %y=2*(L-M).*x.*mask1;
end
end

