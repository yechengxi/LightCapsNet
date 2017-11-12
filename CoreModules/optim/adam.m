function [  net,res,opts ] = adam(  net,res,opts )
% Modified Adam using second-order information.
%   1. Kingma, D., & Ba, J. (2014). 
%   Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
%   2. Ye, C., Yang, Y., Fermuller, C., & Aloimonos, Y. (2017). 
%   On the Importance of Consistency in Training Deep Neural Networks. arXiv preprint arXiv:1708.00631.

    if ~isfield(opts.parameters,'second_order')
        opts.parameters.second_order=0;
    end
    if opts.parameters.second_order
        [  net,res,opts ] = gradient_decorrelation(  net,res,opts );
    end
    
    if ~isfield(opts.parameters,'weightDecay')
        opts.parameters.weightDecay=0;
    end
    
    if (~isfield(opts.parameters,'mom2'))
        opts.parameters.mom2=0.999; 
    end
    
    if ~isfield(opts.parameters,'eps')
        opts.parameters.eps=1e-8;
    end
    
    if ~isfield(net,'iterations')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
        net.iterations=0;
    end
    
    net.iterations=net.iterations+1;
   
    mom_factor=(1-opts.parameters.mom.^net.iterations);
    mom_factor2=(1-opts.parameters.mom2.^net.iterations);
  
     for layer=1:numel(net.layers)
        if isfield(net.layers{layer},'weights')
            if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)||length(net.layers{layer}.momentum)<4
                net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
                net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
                net.layers{layer}.momentum{3}=net.layers{layer}.momentum{1};%initialize
                net.layers{layer}.momentum{4}=net.layers{layer}.momentum{2};%initialize
                
            end
        end
     end
     
    
    for layer=1:numel(net.layers)
        if isfield(net.layers{layer},'weights')
            
            net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}+(1-opts.parameters.mom).*res(layer).dzdw;
            net.layers{layer}.momentum{3}=opts.parameters.mom.*net.layers{layer}.momentum{3}+(1-opts.parameters.mom).*res(layer).dzdw.^2;
            net.layers{layer}.weights{1}=net.layers{layer}.weights{1}-opts.parameters.lr*net.layers{layer}.momentum{1} ...
                ./(net.layers{layer}.momentum{3}.^0.5+opts.parameters.eps) .*mom_factor2^0.5./mom_factor ...
                - opts.parameters.weightDecay * net.layers{layer}.weights{1};
            
            net.layers{layer}.momentum{2}=opts.parameters.mom.*net.layers{layer}.momentum{2}+(1-opts.parameters.mom).*res(layer).dzdb;
            net.layers{layer}.momentum{4}=opts.parameters.mom.*net.layers{layer}.momentum{4}+(1-opts.parameters.mom).*res(layer).dzdb.^2;
            net.layers{layer}.weights{2}=net.layers{layer}.weights{2}-opts.parameters.lr*net.layers{layer}.momentum{2} ...
                ./(net.layers{layer}.momentum{4}.^0.5+opts.parameters.eps) .*mom_factor2^0.5./mom_factor;
            
        end
    end
   
    if ~isfield(opts,'reset_mom')||opts.reset_mom==1
        opts.reset_mom=0;
    end
end

