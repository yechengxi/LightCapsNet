function [  net,res,opts ] = sgd(  net,res,opts )
%Stochastic gradient descent algorithm


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
        opts.parameters.clip=0;
    end
    
    if ~isfield(net,'iterations')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
        net.iterations=0;
    end

    
    net.iterations=net.iterations+1;    
    mom_factor=(1-opts.parameters.mom.^net.iterations);
    
    
    
    for layer=1:numel(net.layers)
        if isfield(net.layers{layer},'weights')&&~isempty(net.layers{layer}.weights)
            
            if opts.parameters.clip>0
                mask=abs(res(layer).dzdw)>opts.parameters.clip;
                res(layer).dzdw(mask)=sign(res(layer).dzdw(mask)).*opts.parameters.clip;%%this type of processing seems to be very helpful
                mask=abs(res(layer).dzdb)>opts.parameters.clip;
                res(layer).dzdb(mask)=sign(res(layer).dzdb(mask)).*opts.parameters.clip;
            end
            if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
                net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
                net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
                
            end
            net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}-(1-opts.parameters.mom).*res(layer).dzdw - opts.parameters.weightDecay * net.layers{layer}.weights{1};
            net.layers{layer}.weights{1}=net.layers{layer}.weights{1}+opts.parameters.lr*net.layers{layer}.momentum{1}./mom_factor;
            
            net.layers{layer}.momentum{2}=opts.parameters.mom.*net.layers{layer}.momentum{2}-(1-opts.parameters.mom).*res(layer).dzdb;
            net.layers{layer}.weights{2}=net.layers{layer}.weights{2}+opts.parameters.lr*net.layers{layer}.momentum{2}./mom_factor;

        end
    end
   
    if ~isfield(opts,'reset_mom')||opts.reset_mom==1
        opts.reset_mom=0;
    end
end

