function [  net,res,opts ] = adagrad(  net,res,opts )
%   Modified Adagrad using second-order information:
%   1. Duchi, J., Hazan, E., & Singer, Y. (2011). 
%   Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul), 2121-2159.
%   2. Ye, C., Yang, Y., Fermuller, C., & Aloimonos, Y. (2017). 
%   On the Importance of Consistency in Training Deep Neural Networks. arXiv preprint arXiv:1708.00631.
%

    if ~isfield(opts.parameters,'second_order')
        opts.parameters.second_order=0;
    end
    if opts.parameters.second_order
        [  net,res,opts ] = gradient_decorrelation(  net,res,opts );
    end
    
    if ~isfield(opts.parameters,'weightDecay')
        opts.parameters.weightDecay=1e-4;
    end
    
    if ~isfield(opts.parameters,'eps')
        opts.parameters.eps=1e-6;
    end
    
    for layer=1:numel(net.layers)
        if isfield(net.layers{layer},'weights')
            
            if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
                net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
                net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
            end
            
            net.layers{layer}.momentum{1}=net.layers{layer}.momentum{1}+res(layer).dzdw.^2;
            net.layers{layer}.weights{1}=net.layers{layer}.weights{1}-opts.parameters.lr*res(layer).dzdw./(net.layers{layer}.momentum{1}.^0.5+opts.parameters.eps)- opts.parameters.weightDecay * net.layers{layer}.weights{1};
            
            net.layers{layer}.momentum{2}=net.layers{layer}.momentum{2}+res(layer).dzdb.^2;
            net.layers{layer}.weights{2}=net.layers{layer}.weights{2}-opts.parameters.lr*res(layer).dzdb./(net.layers{layer}.momentum{2}.^0.5+opts.parameters.eps);

        end
    end
   
    if ~isfield(opts,'reset_mom')||opts.reset_mom==1
        opts.reset_mom=0;
    end
    
end

