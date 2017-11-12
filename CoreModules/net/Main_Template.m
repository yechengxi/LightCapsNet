%%%%%%define your parameters here
%{
%%mnist mlp
opts.dataset_name='mnist';
network_name='cnn'; %network name
use_gpu=1; %use gpu or not
PrepareDataFunc=@PrepareData_MNIST_MLP;
NetInit=@net_init_mlp_mnist;
use_selective_sgd=1;
ssgd_search_freq=10;
learning_method=@sgd; %training method: @sgd,@adagrad,@rmsprop,@adam
sgd_lr=1e-2;
opts.n_epoch=20;
opts.LoadResults=0;
%}


%{
n_epoch=20; %training epochs
dataset_name='mnist'; %dataset name
network_name='cnn'; %network name
use_gpu=1; %use gpu or not 

%function handle to prepare your data
PrepareDataFunc=@PrepareData_MNIST_CNN;
%function handle to initialize the network
NetInit=@net_init_cnn_mnist;

%automatically select learning rates
use_selective_sgd=1; 
%select a new learning rate every n epochs
ssgd_search_freq=10; 
learning_method=@sgd; %training method: @sgd,@adagrad,@rmsprop,@adam

%sgd parameter 
%(unnecessary if selective-sgd is used)
%sgd_lr=5e-2;
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end 


%opts=[];
%addpath('./CoreModules');
opts.n_epoch=n_epoch; %training epochs
opts.dataset_name=dataset_name; %dataset name
opts.network_name=network_name; %network name
opts.use_gpu=use_gpu; %use gpu or not 

if ~isfield(opts,'datatype')
    opts.datatype ='single';
end

if ~isfield(opts,'LoadNet')
    opts.LoadNet=0;
end

opts.dataDir=['./',opts.dataset_name,'/'];
opts=PrepareDataFunc(opts);

%opts.parameters=[];

opts.parameters.current_ep=1;



%%%parameters in the training

opts.parameters.learning_method=learning_method;
opts.parameters.selective_sgd=use_selective_sgd;%call selective-sgd



%selective-sgd parameters
if opts.parameters.selective_sgd==1
    if ~isfield(opts.parameters,'search_iterations')
        opts.parameters.search_iterations=30;%iterations used to determine the learning rate
    end
    opts.parameters.ssgd_search_freq=ssgd_search_freq;%%search every n epoch
    if ~isfield(opts.parameters,'lrs')
        
        opts.parameters.lrs =[1,.5];%initialize selection range
        if ~strcmp(func2str(opts.parameters.learning_method),'sgd')&&~strcmp(func2str(opts.parameters.learning_method),'sgd2')
            opts.parameters.lrs =opts.parameters.lrs.*1e-1;
        end
        opts.parameters.lrs=[opts.parameters.lrs,opts.parameters.lrs*1e-1,opts.parameters.lrs*1e-2,opts.parameters.lrs*1e-3];%initialize selection range
    end
    opts.parameters.selection_count=0;%initialize
    opts.parameters.selected_lr=[];%initialize
end

if opts.parameters.selective_sgd==0
    opts.parameters.lr =sgd_lr;
end


%%sgd parameters
if ~isfield(opts.parameters,'mom')
    opts.parameters.mom =0.9;
end

%adam parameters
if strcmp(func2str(opts.parameters.learning_method),'adam')
    if ~isfield(opts.parameters,'mom2')
        opts.parameters.mom2 =0.999;
    end
end

if ~isfield(opts.parameters,'batch_size')
    opts.parameters.batch_size=500;
end
if ~isfield(opts.parameters,'weightDecay')
    opts.parameters.weightDecay=1e-4;
end

opts=generate_output_filename(opts);


if ~isfield(opts,'plot')
    opts.plot =1;
end

if ~isfield(opts,'LoadResults')
    opts.LoadResults=0;
end

if ~opts.LoadResults
    TrainingScript();
end
