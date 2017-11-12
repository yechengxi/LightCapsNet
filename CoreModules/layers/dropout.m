function [y,mask] = dropout(x,dzdy,opts)

% determine mask
mask = opts.mask ;

if isempty(dzdy)
    scale = 1 / (1 - opts.rate);
    mask =  (rand(size(x), 'like', x) >= opts.rate) .* scale;         
end

if isempty(dzdy)
    y = x.*mask  ;   
else
    y = dzdy.*mask ;   
end
