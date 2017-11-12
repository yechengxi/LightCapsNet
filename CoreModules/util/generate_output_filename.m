function [ opts ] = generate_output_filename( opts )
%GENERATE_OUTPUT_FILENAME Summary of this function goes here
%   Detailed explanation goes here

    if ~isfield(opts,'network_name')
       opts.network_name='net'; 
    end
    opts.output_name=[opts.dataset_name,'-',opts.network_name];

    if opts.parameters.selective_sgd==1
        opts.output_name=[opts.output_name,'-ssgd-search-freq-',num2str(opts.parameters.ssgd_search_freq)];          
    end
    opts.output_name=[opts.output_name,'-',func2str(opts.parameters.learning_method)];

    if opts.parameters.selective_sgd==0  
        opts.output_name=[opts.output_name,'-lr-',num2str(opts.parameters.lr)];
    end

    opts.saved_filenames=[opts.output_name,'-epoch-*.mat'];    

    opts.output_dir=['./',opts.dataset_name,'-tests/temp/'];
    opts.output_name=[opts.output_name,'-epoch-'];
    
    if ~exist(opts.output_dir,'dir')
        mkdir(opts.output_dir)
    end
    

end

