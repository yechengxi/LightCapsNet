function y = sigmoid_ln(x,dzdy)
    
    y = 1 ./ (1 + exp(-x));

    if nargin > 1 &&~isempty(dzdy)
      y = dzdy .* (y .* (1 - y)) ;
    end

end
