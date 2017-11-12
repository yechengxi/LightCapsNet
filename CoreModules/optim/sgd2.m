function [  net,res,opts ] = sgd2(  net,res,opts )
% Stochastic gradient descent using second-order information.
%   Ye, C., Yang, Y., Fermuller, C., & Aloimonos, Y. (2017). 
%   On the Importance of Consistency in Training Deep Neural Networks. arXiv preprint arXiv:1708.00631.

    if ~isfield(opts.parameters,'weightDecay')
        opts.parameters.weightDecay=1e-4;
    end    
    if ~isfield(opts.parameters,'lambda_sgd2')
        opts.parameters.lambda_sgd2=1e0;
    end
    if ~isfield(opts.parameters,'large_matrix_inversion')
        opts.parameters.large_matrix_inversion=0;
    end
    if ~isfield(opts.parameters,'max_inv_size')
        opts.parameters.max_inv_size=500;
    end    
    if ~isfield(opts.parameters,'clip')
        opts.parameters.clip=0;
    end
    if ~isfield(opts.parameters,'decorr_bias')
        opts.parameters.decorr_bias=1;
    end
    if ~isfield(net,'iterations')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
        net.iterations=0;
    end
   
    net.iterations=net.iterations+1;
    
    mom_factor=(1-opts.parameters.mom.^net.iterations);
    max_inv_size=opts.parameters.max_inv_size;
    lambda=opts.parameters.lambda_sgd2;
    
    
    for layer=1:numel(net.layers)
        if isfield(net.layers{layer},'weights')&&~isempty(net.layers{layer}.weights)
            if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
                net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
                net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
            end
            
            dzdw=res(layer).dzdw;
            dzdb=res(layer).dzdb;
            
            if length(net.layers{layer}.weights)==2
                x=res(layer).x;
                batch_dim=length(size(x));%This assumes the batch size must be >1 
                if batch_dim==4%2d cnn
                    x=permute(x,[3,1,2,4]);x=reshape(x,size(x,1),[]);
                    dzdw=permute(dzdw,[1,2,4,3]);new_size=size(dzdw);dzdw=reshape(dzdw,prod(new_size(1:3)),new_size(4));
                    K=size(dzdw,1)/numel(dzdb);dzdb=repelem(dzdb(:),K,1);
                end
                if batch_dim==3%1d cnn
                    x=permute(x,[2,1,3]);x=reshape(x,size(x,1),[]);
                    dzdw=permute(dzdw,[1,3,2]);new_size=size(dzdw);dzdw=reshape(dzdw,prod(new_size(1:2)),new_size(3));
                    K=size(dzdw,1)/numel(dzdb);dzdb=repelem(dzdb(:),K,1);
                end
                subsample=1;batch_size=size(x,2);
                if batch_size>1e4,subsample=ceil(min(50,batch_size/1e4));end
                if subsample>1,x=x(:,1:subsample:end);end
                if opts.parameters.decorr_bias==1
                    %insert bias
                    x=[ones(1,size(x,2),'like',x);x];
                    dzdw=[dzdb,dzdw];
                end
                if size(dzdw,2)<=max_inv_size %small scale inversion
                    dzdw=dzdw/(x*x'./size(x,2)+lambda*eye(size(x,1),'like',x));  
                elseif opts.parameters.large_matrix_inversion %divide large scale into smaller scale
                    order=randperm(size(dzdw,2));
                    for i=1:max_inv_size:length(order) %could have been parallelized 
                       block_size=min(max_inv_size,length(order)-i+1);
                       idx=order(i:i+block_size-1);x_tmp=x(idx,:);
                       dzdw(:,idx)=dzdw(:,idx)/(x_tmp*x_tmp'./size(x_tmp,2)+lambda*eye(size(x_tmp,1),'like',x));
                    end
                end
                if opts.parameters.decorr_bias==1
                    dzdb=dzdw(:,1);dzdw(:,1)=[];
                end
                if batch_dim==4,dzdw=reshape(dzdw,new_size);dzdw=permute(dzdw,[1,2,4,3]);end
                if batch_dim==3,dzdw=reshape(dzdw,new_size);dzdw=permute(dzdw,[1,3,2]);end
                if batch_dim>2%for cnn:                      
                    %dzdb is decorrelated with dzdw, take average to smooth the results.                    
                    dzdb=reshape(mean(reshape(dzdb(:),K,[]),1),size(res(layer).dzdb));
                end
            end
            
            if opts.parameters.clip>0
                mask=abs(res(layer).dzdw)>opts.parameters.clip;
                res(layer).dzdw(mask)=sign(res(layer).dzdw(mask)).*opts.parameters.clip;%%this type of processing seems to be very helpful
                mask=abs(res(layer).dzdb)>opts.parameters.clip;
                res(layer).dzdb(mask)=sign(res(layer).dzdb(mask)).*opts.parameters.clip;
            end

            
            %sgd updates
            net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}-(1-opts.parameters.mom).*dzdw - opts.parameters.weightDecay * net.layers{layer}.weights{1};
            net.layers{layer}.weights{1}=net.layers{layer}.weights{1}+opts.parameters.lr*net.layers{layer}.momentum{1}./mom_factor;
            
            net.layers{layer}.momentum{2}=opts.parameters.mom.*net.layers{layer}.momentum{2}-(1-opts.parameters.mom).*dzdb;
            net.layers{layer}.weights{2}=net.layers{layer}.weights{2}+opts.parameters.lr*net.layers{layer}.momentum{2}./mom_factor;

        end
    end
   
    if ~isfield(opts,'reset_mom')||opts.reset_mom==1
        opts.reset_mom=0;
    end
end

