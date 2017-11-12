function net = SwitchProcessor(net, hardware)
switch hardware
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown destination ''%s''.', destination) ;
end

for l=1:numel(net.layers)
  switch net.layers{l}.type
    case {'conv','mlp'}
      for f = {'weights', 'momentum','lr'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          for j=1:numel(net.layers{l}.(f))
            net.layers{l}.(f){j} = moveop(net.layers{l}.(f){j}) ;
          end
        end
      end
      otherwise
  end
end
