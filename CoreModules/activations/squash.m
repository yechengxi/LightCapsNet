function [y] = squash(x,d,dzdy)
%SQUASH Summary of this function goes here
%   Detailed explanation goes here
batch_dim=length(size(x));% This assumes the batch size must be >1 

if batch_dim>2%2d cnn
    x=permute(x,[3,1,2,4]);
end

if isempty(dzdy)
    
    
    y=reshape(x,d,[]);
    L2=sum(y.^2,1);
    y=L2.^0.5./(1+L2).*y;
    y=reshape(y,size(x));
    
    if batch_dim==4%2d cnn
       y=permute(y,[2,3,1,4]);
    end
    
else
    
    if batch_dim>2%2d cnn
        dzdy=permute(dzdy,[3,1,2,4]);
    end

    y=reshape(x,d,[]);             % capsule_dim x batch
    dzdy=reshape(dzdy,d,[]);       % capsule_dim x batch
    L2=sum(reshape(x,d,[]).^2,1);  %           1 x batch
    % this line looks a bit crazy.
    y=(1./(L2.^0.5.*(1+L2))-2*L2.^0.5./((1+L2).^2)).*y.*sum(y.*dzdy,1) ...
        +L2.^0.5./(1+L2).*dzdy;
    y=reshape(y,size(x));
    
    if batch_dim==4
       y=permute(y,[2,3,1,4]);
    end
end

end

