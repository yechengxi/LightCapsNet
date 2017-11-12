function [ net,y,opts] = rmsnorm( net, x,layer_idx, dzdy,opts )

    if ~isfield(net,'iterations'),net.iterations=0;end
    if ~isfield(opts.parameters, 'mom_rmsnorm'),opts.parameters.mom_rmsnorm=0.1;end
    if ~isfield(opts.parameters, 'eps_rmsnorm'),opts.parameters.eps_rmsnorm=1e-2;end
    %
    if layer_idx>1
        eps_rmsnorm=opts.parameters.eps_rmsnorm;
    else
        eps_rmsnorm=1e-10;% the first layer is a standard data normalization
    end
    
    if ~isfield(net,'iterations_rmsnorm')
        net.iterations_rmsnorm=0; 
    else
        net.iterations_rmsnorm=net.iterations;
    end
    
    mom_factor=1-opts.parameters.mom_rmsnorm.^(net.iterations_rmsnorm+1);
    batch_dim=length(size(x));% This assumes the batch size must be >1 
    shape_x=size(x);
    if batch_dim==4%2d cnn
        x=permute(x,[3,1,2,4]);x=reshape(x,size(x,1),[]);
    end
    if batch_dim==3%1d cnn 
       x=permute(x,[2,1,3]);x=reshape(x,size(x,1),[]);
    end
    
    [d,batch_size]=size(x);
 
    if ~isfield(net.layers{1,layer_idx},'denom2')
        net.layers{1,layer_idx}.denom2=zeros(d,1,'like',x);
    end
    
    if(opts.training&&isempty(dzdy))
       
        denom2=mean(x.^2,2);
        net.layers{1,layer_idx}.denom2=opts.parameters.mom_rmsnorm*net.layers{1,layer_idx}.denom2+(1-opts.parameters.mom_rmsnorm)*denom2;
        
    end
    
    eps=max(net.layers{1,layer_idx}.denom2(:)+1e-6)*eps_rmsnorm;
    
    denom=(net.layers{1,layer_idx}.denom2./mom_factor+eps).^0.5;      
    
    if(isempty(dzdy))
        y = x./denom;        
    else
        if batch_dim==4
            dzdy=permute(dzdy,[3,1,2,4]);dzdy=reshape(dzdy,size(dzdy,1),[]);
        end
        if batch_dim==3 
            dzdy=permute(dzdy,[2,1,3]);dzdy=reshape(dzdy,size(dzdy,1),[]);
        end
        y = dzdy./denom; %simple update rule
        
    end
    
    if batch_dim==4
       y=reshape(y,[shape_x(3),shape_x(1),shape_x(2),shape_x(4)]);   
       y=permute(y,[2,3,1,4]);
    end
    if batch_dim==3 
        y=reshape(y,shape_x(2),shape_x(1),shape_x(3));
        y=permute(y,[2,1,3]);
    end 
end

