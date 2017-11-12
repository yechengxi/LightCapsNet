function [ net,opts ] = selective_sgd( net,opts )
%NET_APPLY_SELECTIVE_SGD Summary of this function goes here
%   Detailed explanation goes here


    if (mod(opts.parameters.current_ep,opts.parameters.ssgd_search_freq)==1||opts.parameters.ssgd_search_freq==1)

        if ~isfield(opts.parameters,'selection_count')
            opts.parameters.selection_count=0;
        end

        
        [lr_best]=select_learning_rate(net,opts);
        opts.parameters.lr=lr_best;

            
        if ~isfield(opts.parameters,'selected_lr')
            opts.parameters.selected_lr(1)=lr_best;
        else        
            opts.parameters.selected_lr(end+1)=lr_best;
        end

        
        opts.parameters.selection_count=opts.parameters.selection_count+1;
        
        disp(['Selected learning rate: ',num2str(opts.parameters.lr)]);

    end
        
end

