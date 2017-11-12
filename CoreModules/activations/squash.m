function [y] = squash(x,d,dzdy)
%SQUASH Summary of this function goes here
%   Detailed explanation goes here
if isempty(dzdy)
    y=reshape(x,d,[]);
    L2=sum(y.^2,1);
    y=L2.^0.5./(1+L2).*y;
    y=reshape(y,size(x));    
else
    y=reshape(x,d,[]);
    dzdy=reshape(dzdy,d,[]);
    L2=sum(reshape(x,d,[]).^2,1);
    % this line looks a bit crazy.
    y=(1./(L2.^0.5.*(1+L2))-2*L2.^0.5./((1+L2).^2)).*y.*sum(y.*dzdy,1) ...
        +L2.^0.5./(1+L2).*dzdy;
    y=reshape(y,size(x));
end

end

