function y = leaky_relu(x,dzdy,p)

if ~exist('p','var')|| isempty(p)
    p=0.5;
end
  
if nargin <= 1 || isempty(dzdy)
    y = max(x, single(0)) + p*min(x, single(0));
else
    y = dzdy .* (x > single(0))+dzdy .* p.*(x < single(0)) ;
end
  
end
