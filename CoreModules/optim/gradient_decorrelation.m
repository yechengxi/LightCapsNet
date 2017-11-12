function [  net,res,opts ] = gradient_decorrelation(  net,res,opts )
% Decorrelating gradient descents using second-order information.
%   Ye, C., Yang, Y., Fermuller, C., & Aloimonos, Y. (2017). 
%   On the Importance of Consistency in Training Deep Neural Networks. arXiv preprint arXiv:1708.00631.

    if ~isfield(opts.parameters,'lambda_sgd2')
        opts.parameters.lambda_sgd2=1e0;
    end
    if ~isfield(opts.parameters,'large_matrix_inversion')
        opts.parameters.large_matrix_inversion=0;
    end
    if ~isfield(opts.parameters,'max_inv_size')
        opts.parameters.max_inv_size=500;
    end    
    if ~isfield(opts.parameters,'decorr_bias')
        opts.parameters.decorr_bias=1;
    end
    
    max_inv_size=opts.parameters.max_inv_size;
    lambda=opts.parameters.lambda_sgd2;
    
    
    for layer=1:numel(net.layers)
        if isfield(net.layers{layer},'weights')&&~isempty(net.layers{layer}.weights)            
            
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
               res(layer).dzdw=dzdw;
               res(layer).dzdb=dzdb;
            end
            
        end
    end
   
end

