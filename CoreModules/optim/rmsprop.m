function [  net,res,opts ] = rmsprop(  net,res,opts )
% Modified RMSProp using second-order information.   
%   1.Tieleman, T. and Hinton, G. Lecture 6.5 - RMSProp, COURSERA: Neural Networks for Machine Learning.
%   Technical report, 2012.
%   2.Ye, C., Yang, Y., Fermuller, C., & Aloimonos, Y. (2017). 
%   On the Importance of Consistency in Training Deep Neural Networks. arXiv preprint arXiv:1708.00631.

    if ~isfield(opts.parameters,'second_order')
        opts.parameters.second_order=0;
    end
    if opts.parameters.second_order
        [  net,res,opts ] = gradient_decorrelation(  net,res,opts );
    end

    if ~isfield(opts.parameters,'weightDecay')
        opts.parameters.weightDecay=1e-4;
    end

    
    if ~isfield(opts.parameters,'clip')
        opts.parameters.clip=1e0;
    end
    
    if ~isfield(opts.parameters,'eps')
        opts.parameters.eps=1e-6;
    end
    
    if ~isfield(net,'iterations')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
        net.iterations=0;
    end
    
    net.iterations=net.iterations+1;
    
    mom_factor=(1-opts.parameters.mom.^net.iterations);
    
    for layer=1:numel(net.layers)
        if isfield(net.layers{layer},'weights')
            if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
                net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
                net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
                
            end
            
            net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}+(1-opts.parameters.mom).*(res(layer).dzdw.^2);
            normalized_grad=res(layer).dzdw./(net.layers{layer}.momentum{1}.^0.5+opts.parameters.eps)./mom_factor;
            if isfield(opts.parameters,'clip')&&opts.parameters.clip>0
                mask=abs(normalized_grad)>opts.parameters.clip;
                normalized_grad(mask)=sign(normalized_grad(mask)).*opts.parameters.clip;
            end
            net.layers{layer}.weights{1}=net.layers{layer}.weights{1}-opts.parameters.lr*normalized_grad- opts.parameters.weightDecay * net.layers{layer}.weights{1};
            
            net.layers{layer}.momentum{2}=opts.parameters.mom.*net.layers{layer}.momentum{2}+(1-opts.parameters.mom).*(res(layer).dzdb.^2);
            normalized_grad=res(layer).dzdb./(net.layers{layer}.momentum{2}.^0.5+opts.parameters.eps)./mom_factor;
            if isfield(opts.parameters,'clip')&&opts.parameters.clip>0
                mask=abs(normalized_grad)>opts.parameters.clip;
                normalized_grad(mask)=sign(normalized_grad(mask)).*opts.parameters.clip;
            end
            net.layers{layer}.weights{2}=net.layers{layer}.weights{2}-opts.parameters.lr*normalized_grad;
        end
    end
    
   if ~isfield(opts,'reset_mom')||opts.reset_mom==1
        opts.reset_mom=0;
    end
end

