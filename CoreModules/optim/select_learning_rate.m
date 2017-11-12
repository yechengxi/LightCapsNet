function [lr_best,min_loss] = select_learning_rate(net,opts )
%   Detailed explanation goes here
   % do the grid search   
   
   if(opts.use_gpu)       
        for i=1:length(net)
            net(i)=SwitchProcessor(net(i),'cpu');
        end
   end
   temp_net=net;
   net=[];
   opts.parameters.loss=zeros(size(opts.parameters.lrs));

    for l=1:length(opts.parameters.lrs)%learning rate list

        net=temp_net;
        if(opts.use_gpu)       
            for i=1:length(net)
                net(i)=SwitchProcessor(net(i),'gpu');
            end
        end

        
        opts.parameters.lr=opts.parameters.lrs(l);%test the candidate learning rate

        if (isfield(opts.parameters,'selected_lr') && length(opts.parameters.selected_lr)>0 && opts.parameters.lr>opts.parameters.selected_lr(1)) 

            loss=10000;
            opts.parameters.loss(l) =loss;
            continue;

        else
            
            batch_dim=length(size(opts.train));
            idx_nd=repmat({':'},[1,batch_dim]);
                 
            for mini_b=1:min(opts.n_train_batch,opts.parameters.search_iterations)

                idx=opts.order(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);
                idx_nd{end}=idx;
                opts.idx_nd=idx_nd;
                opts.idx=idx;
                res(1).x=opts.train(idx_nd{:});
                
                %classification
                if strcmp(net.layers{end}.type,'softmaxloss')
                    res(1).class=opts.train_labels(idx);
                end


                %forward
                [ net,res,opts ] = net_ff( net,res,opts );
    
            
                %%%%backward
                opts.dzdy=1.0;
                
                
                [ net,res,opts ] = net_bp( net,res,opts );

                %%collect stats

                loss=gather(mean(res(end).x(:)));

                [net,res,opts] = opts.parameters.learning_method(net,res,opts);
                
            end
            clear net;            

        end

        opts.parameters.loss(l) =loss;
        disp(['Learning rate: ',num2str(opts.parameters.lrs(l)),' loss: ' num2str(loss)]);

    end

    [min_loss,min_idx]=min(opts.parameters.loss);

    lr_best=opts.parameters.lrs(min_idx);
            
end

