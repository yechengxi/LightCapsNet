function [net,opts]=train_net(net,opts)

    if ~isfield(opts,'datatype')
        opts.datatype='single';
    end
    
    opts.training=1;

    if ~isfield(opts.parameters,'learning_method')
        opts.parameters.learning_method='sgd';        
    end
    
    if ~isfield(opts,'display_msg')
        opts.display_msg=1; 
    end
    opts.TrainMiniBatchError=[];
    opts.TrainMiniBatchError_Top5=[];
    opts.TrainMiniBatchLoss=[];
    
    
    tic
    
    opts.order=randperm(opts.n_train);    

    if opts.parameters.selective_sgd==1 
        [ net,opts ] = selective_sgd( net,opts );
    end
    
    batch_dim=length(size(opts.train));
    idx_nd=repmat({':'},[1,batch_dim]);
    
    for mini_b=1:opts.n_train_batch
                
        idx=opts.order(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);       
        idx_nd{end}=idx;
        opts.idx_nd=idx_nd;
        opts.idx=idx;
        res(1).x=opts.train(idx_nd{:});
        
        %classification
        if (strcmp(net.layers{end}.type,'softmaxloss')||strcmp(net.layers{end}.type,'marginloss'))
            res(1).class=opts.train_labels(idx);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%forward%%%%%%%%%%%%%%%%%%%
        [ net,res,opts ] = net_ff( net,res,opts );    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%backward%%%%%%%%%%%%%%%%
        opts.dzdy=1.0;
        
        [ net,res,opts ] = net_bp( net,res,opts );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%summarize the current batch
        loss=double(gather(mean(res(end).x(:))));
        
        if (strcmp(net.layers{end}.type,'softmaxloss')||strcmp(net.layers{end}.type,'marginloss'))
            if strcmp(net.layers{end}.type,'softmaxloss')
                err=error_multiclass(res(1).class,res);
            else
                err=error_multiclass(res(1).class,res,size(res(end-1).x,1)./opts.n_class);
            end
            opts.TrainMiniBatchError=[opts.TrainMiniBatchError;err(1)/opts.parameters.batch_size];
            opts.TrainMiniBatchError_Top5=[opts.TrainMiniBatchError_Top5;err(2)/opts.parameters.batch_size];
            if opts.display_msg==1
                disp(['Minibatch loss: ', num2str(loss),...
                    ', top 1 err: ', num2str(opts.TrainMiniBatchError(end)),...
                    ',top 5 err:,',num2str(opts.TrainMiniBatchError_Top5(end))])
            end
        end
        
        opts.TrainMiniBatchLoss=[opts.TrainMiniBatchLoss;loss];                 
        if (~isfield(opts.parameters,'iterations'))
            opts.parameters.iterations=0; 
        end
        opts.parameters.iterations=opts.parameters.iterations+1;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%stochastic gradients descent%%%%%%%%%%%%%%%%%%%%%%%%
        [ net,res,opts ] = opts.parameters.learning_method( net,res,opts );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    %%summarize the current epoch
     if ~isfield(opts,'results')||~isfield(opts.results,'TrainEpochLoss')
        opts.results.TrainEpochLoss=[];
        opts.results.TrainEpochError=[];
        opts.results.TrainEpochError_Top5=[];
     end
        
    opts.results.TrainEpochLoss=[opts.results.TrainEpochLoss;mean(opts.TrainMiniBatchLoss(:))];

    if (strcmp(net.layers{end}.type,'softmaxloss')||strcmp(net.layers{end}.type,'marginloss'))        
        opts.results.TrainEpochError=[opts.results.TrainEpochError;mean(opts.TrainMiniBatchError(:))];
        opts.results.TrainEpochError_Top5=[opts.results.TrainEpochError_Top5;mean(opts.TrainMiniBatchError_Top5(:))];
        disp(['Epoch ',num2str(opts.parameters.current_ep),...
         ', training loss: ', num2str(opts.results.TrainEpochLoss(end)),...
                ', top 1 err: ', num2str(opts.results.TrainEpochError(end)),...
                ',top 5 err:,',num2str(opts.results.TrainEpochError_Top5(end))])                

    end

    
    if opts.RecordStats==1
        if ~isfield(opts,'results')||~isfield(opts.results,'TrainMiniBatchLoss')
            opts.results.TrainMiniBatchLoss=[];
            opts.results.TrainMiniBatchError=[];
            opts.results.TrainMiniBatchError_Top5=[];
        end
        opts.results.TrainMiniBatchLoss=[opts.results.TrainLoss;opts.TrainMiniBatchLoss];
        opts.results.TrainMiniBatchError=[opts.results.TrainError;opts.TrainMiniBatchError]; 
        opts.results.TrainMiniBatchError_Top5=[opts.results.TrainError_Top5;opts.TrainMiniBatchError_Top5]; 
        
    end
    
    toc;

end




