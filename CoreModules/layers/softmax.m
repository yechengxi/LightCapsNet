function y =softmax(x,dzdy)

if(length(size(x))>2),idx_c=3;end%cnn
if(length(size(x))<=2),idx_c=1;end%mlp

E = exp(x- max(x,[],idx_c)) ;
S = sum(E,idx_c) ;

y =  E./ S ;

if isempty(dzdy), return ; end

% backward
y = y .* (dzdy-sum(dzdy .* y, idx_c));

