function [ net,y,dzdw,dzdb,opts ] = bnorm( net,x,layer_idx,dzdy,opts )
%BNORM Summary of this function goes here
%   Detailed explanation goes here
    dzdw=[];
    dzdb=[];
    if ~isfield(net,'iterations')
        net.iterations=0;
    end
    if ~isfield(net,'iterations_bn')
        net.iterations_bn=0; 
    else
        net.iterations_bn=net.iterations;
    end
    
    if ~isfield(opts.parameters, 'eps_bn')
        opts.parameters.eps_bn=1e-2;
    end
    
    if ~isfield(opts.parameters, 'mom_bn')
        opts.parameters.mom_bn=0.1;% this one needs to be small
    end
        
    if ~isfield(opts.parameters, 'simple_bn')
        opts.parameters.simple_bn=0;
    end
    
    batch_dim=length(size(x));%% This assumes the batch size must be >1 
    shape_x=size(x);
    
    if batch_dim==4
       x=permute(x,[3,1,2,4]);x=reshape(x,size(x,1),[]);
    end
    if batch_dim==3 
       x=permute(x,[2,1,3]);x=reshape(x,size(x,1),[]);
    end

    if ~isfield(net.layers{1,layer_idx},'weights')        
        sz=[shape_x(end-1),1];
        net.layers{1,layer_idx}.weights{1}=ones(sz,'like',x);
        net.layers{1,layer_idx}.weights{2}=zeros(sz,'like',x);
        for i=1:2
            net.layers{1,layer_idx}.momentum{i}=zeros(sz,'like',x);
        end        
    end
    
    if (opts.training&&isempty(dzdy))
        if(net.iterations_bn==0||(isfield(opts,'reset_mom')&&opts.reset_mom==1)||length(net.layers{1,layer_idx}.weights)<4)  
            for i=3:4
                net.layers{1,layer_idx}.weights{i}=0;
            end
        end
    end    
    mom_factor=1-opts.parameters.mom_bn.^(net.iterations_bn+1);
    
    if(opts.training&&isempty(dzdy))
        net.layers{1,layer_idx}.weights{3}=opts.parameters.mom_bn*net.layers{1,layer_idx}.weights{3}+(1-opts.parameters.mom_bn)*mean(x,2);   
    end
    
    mu=net.layers{1,layer_idx}.weights{3}./mom_factor;
    x=x-mu;
    
    if(opts.training&&isempty(dzdy))
        net.layers{1,layer_idx}.weights{4}=opts.parameters.mom_bn*net.layers{1,layer_idx}.weights{4}+(1-opts.parameters.mom_bn)*mean(x.^2,2);  
    end
    
    sigma=(net.layers{1,layer_idx}.weights{4}./mom_factor+opts.parameters.eps_bn).^0.5;
    
    if(isempty(dzdy))
        y=x.*(net.layers{1,layer_idx}.weights{1}./sigma);
        y=y+net.layers{1,layer_idx}.weights{2};        
    
    else        
        if batch_dim==4
            dzdy=permute(dzdy,[3,1,2,4]);dzdy=reshape(dzdy,size(dzdy,1),[]);
        end
        if batch_dim==3 
            dzdy=permute(dzdy,[2,1,3]);dzdy=reshape(dzdy,size(dzdy,1),[]);
        end
        
        x=x./sigma;            
        
        dzdw=sum(dzdy.*x,2)./shape_x(end);
        dzdb=sum(dzdy,2)./shape_x(end);
        denom=size(x,2)./shape_x(end);
        if ~opts.parameters.simple_bn
            %the complicated version
            x=x.*(dzdw./denom);
            x=x+dzdb./denom;
            
            dzdy=(dzdy-x.*(1-opts.parameters.mom_bn));
            %[max(abs(dzdy(:))), max(abs(tmp(:)))]
        end
        
        %the simple version:
        y=dzdy.*(net.layers{1,layer_idx}.weights{1}./sigma);
    end
    
    if batch_dim==4
        y=reshape(y,shape_x(3),shape_x(1),shape_x(2),shape_x(4));
        y=permute(y,[2,3,1,4]);
    end
    if batch_dim==3 
        y=reshape(y,shape_x(2),shape_x(1),shape_x(3));
        y=permute(y,[2,1,3]);
    end        
end
