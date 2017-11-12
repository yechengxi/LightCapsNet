function [opts]=test_net(net,opts)

    opts.training=0;

    opts.TestMiniBatchError=[];
    opts.TestMiniBatchError_Top5=[];
    opts.TestMiniBatchLoss=[];
  
    
    if ~isfield(opts,'validating')
        opts.validating=0;
    end
    
    if opts.validating
        n_batch=opts.n_valid_batch;
    else
        n_batch=opts.n_test_batch;
    end
    
    batch_dim=length(size(opts.test));
    idx_nd=repmat({':'},[1,batch_dim]);
    
    for mini_b=1:n_batch
        
        idx=1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size;
        idx_nd{end}=idx;
        opts.idx_nd=idx_nd;
        opts.idx=idx;
        
        if opts.validating
            %input
            res(1).x=opts.valid(idx_nd{:});
            %output
            %classification
            if (strcmp(net.layers{end}.type,'softmaxloss')||strcmp(net.layers{end}.type,'marginloss'))
                res(1).class=opts.valid_labels(idx);
            end
        else
            %input
            res(1).x=opts.test(idx_nd{:});
            %output
            %classification
            if (strcmp(net.layers{end}.type,'softmaxloss')||strcmp(net.layers{end}.type,'marginloss'))
                res(1).class=opts.test_labels(idx);
            end
        end
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%forward%%%%%%%%%%%%%%%%%%%
        [ net,res,opts ] = net_ff( net,res,opts );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        loss=double(gather(mean(res(end).x(:))));
      
        opts.TestMiniBatchLoss=[opts.TestMiniBatchLoss;loss];
        
        if (strcmp(net.layers{end}.type,'softmaxloss')||strcmp(net.layers{end}.type,'marginloss'))
            if strcmp(net.layers{end}.type,'softmaxloss')
                err=error_multiclass(res(1).class,res);
            else
                err=error_multiclass(res(1).class,res,size(squeeze(res(end-1).x),1)./opts.n_class);
            end
            opts.TestMiniBatchError=[opts.TestMiniBatchError;err(1)/opts.parameters.batch_size];
            opts.TestMiniBatchError_Top5=[opts.TestMiniBatchError_Top5;err(2)/opts.parameters.batch_size];
        end
        
       
      
    end

    
    if opts.validating
        if ~isfield(opts,'results')||~isfield(opts.results,'ValidEpochLoss')
            opts.results.ValidEpochLoss=[];
            opts.results.ValidEpochError=[];
            opts.results.ValidEpochError_Top5=[];
          
        end
        opts.results.ValidEpochLoss=[opts.results.ValidEpochLoss;mean(opts.TestMiniBatchLoss(:))];
        if (strcmp(net.layers{end}.type,'softmaxloss')||strcmp(net.layers{end}.type,'marginloss'))
            opts.results.ValidEpochError=[opts.results.ValidEpochError;mean(opts.TestMiniBatchError(:))];
            opts.results.ValidEpochError_Top5=[opts.results.ValidEpochError_Top5;mean(opts.TestMiniBatchError_Top5(:))];
             disp(['Epoch ',num2str(opts.parameters.current_ep),...
                 ', validation loss: ', num2str(opts.results.ValidEpochLoss(end)),...
                        ', top 1 err: ', num2str(opts.results.ValidEpochError(end)),...
                        ',top 5 err:,',num2str(opts.results.ValidEpochError_Top5(end))])                
        end
        
    
    else
        if ~isfield(opts,'results')||~isfield(opts.results,'TestEpochLoss')
        
            opts.results.TestEpochLoss=[];
            opts.results.TestEpochError=[];
            opts.results.TestEpochError_Top5=[];
    
        end        
        opts.results.TestEpochLoss=[opts.results.TestEpochLoss;mean(opts.TestMiniBatchLoss(:))];
        if (strcmp(net.layers{end}.type,'softmaxloss')||strcmp(net.layers{end}.type,'marginloss'))
            opts.results.TestEpochError=[opts.results.TestEpochError;mean(opts.TestMiniBatchError(:))];
            opts.results.TestEpochError_Top5=[opts.results.TestEpochError_Top5;mean(opts.TestMiniBatchError_Top5(:))];
             disp(['Epoch ',num2str(opts.parameters.current_ep),...
                 ', testing loss: ', num2str(opts.results.TestEpochLoss(end)),...
                        ', top 1 err: ', num2str(opts.results.TestEpochError(end)),...
                        ',top 5 err:,',num2str(opts.results.TestEpochError_Top5(end))])                
        end
        
    
    end
    
    
end


