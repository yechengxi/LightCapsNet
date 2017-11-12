function y = modu(x,dzdy)
%ModU activation function
%Ye, C., Yang, Y., Fermuller, C., & Aloimonos, Y. (2017). 
%On the Importance of Consistency in Training Deep Neural Networks. arXiv preprint arXiv:1708.00631.

  if nargin <= 1 || isempty(dzdy)
    y = abs(x) ;
  else
    y = dzdy .* sign(x) ;
  end
  
end
