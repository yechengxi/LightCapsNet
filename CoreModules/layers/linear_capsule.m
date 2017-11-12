function [y, dzdw,dzdb,opts] = linear_capsule(x,weight,bias,d,dzdy,opts)
%LINEAR_CAPSULE Summary of this function goes here
%   Detailed explanation goes here
dzdw=[];  
dzdb=[];  
if isempty(dzdy)%ff
    y=weight*x;
    if ~isempty(bias)
        y=y+bias;        
    end
else%bp    
    y=weight'*dzdy;    
    if ~isempty(bias)
        dzdb=mean(dzdy,2);%minibatch averaging    
    end    
    dzdy=permute(dzdy,[1,3,2]);
    x=permute(x,[3,1,2]);
    dzdw=dzdy.*x;
    dzdw=mean(dzdw,3);
end

end

