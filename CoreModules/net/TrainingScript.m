
files=dir([fullfile(opts.output_dir,opts.saved_filenames)]);

if opts.LoadNet && length(files)>1
    [~,last_file]=sort([files(:).datenum],'descend');
    if length(files)<opts.n_epoch,end  
    load(fullfile(opts.output_dir,files(last_file(1)).name));
    opts.parameters=parameters;
    opts.results=results;
    
    opts.parameters.current_ep=opts.parameters.current_ep+1;
end

if opts.LoadNet==0 || length(files)==0      
    net=NetInit(opts);
    opts.results=[];

end


opts.RecordStats=0;
opts.n_train_batch=floor(opts.n_train/opts.parameters.batch_size);
if isfield(opts,'n_valid')
    opts.n_valid_batch=floor(opts.n_valid/opts.parameters.batch_size);
end
opts.n_test_batch=floor(opts.n_test/opts.parameters.batch_size);


if(opts.use_gpu)       
    for i=1:length(net)
        net(i)=SwitchProcessor(net(i),'gpu');
    end
else
    for i=1:length(net)
        net(i)=SwitchProcessor(net(i),'cpu');
    end
end

start_ep=opts.parameters.current_ep;
if opts.plot
    figure1=figure;
end
for ep=start_ep:opts.n_epoch
    

    [net,opts]=train_net(net,opts);  
    if isfield(opts,'valid')&&(numel(opts.valid)>0)
        opts.validating=1;
        [opts]=test_net(net,opts);
    end
    opts.validating=0;
    [opts]=test_net(net,opts);
    
    if opts.plot
        %figure(figure1);
        if strcmp(net.layers{end}.type,'softmaxloss')
            subplot(1,2,1); 
            plot(opts.results.TrainEpochError,'b','DisplayName','Train (top1)');hold on;
            plot(opts.results.TrainEpochError_Top5,'b--','DisplayName','Train (top5)');hold on;
            if isfield(opts,'valid')&&(numel(opts.valid)>0)
                plot(opts.results.ValidEpochError,'g','DisplayName','Valid (top1)');hold on
                plot(opts.results.ValidEpochError_Top5,'g--','DisplayName','Valid (top5)');hold on;
            end
            plot(opts.results.TestEpochError,'r','DisplayName','Test (top1)');hold on;
            plot(opts.results.TestEpochError_Top5,'r--','DisplayName','Test (top5)');hold off;
            
            title('Error Rate per Epoch');legend('show');
            subplot(1,2,2); 
            plot(opts.results.TrainEpochLoss,'b','DisplayName','Train');hold on;
            if isfield(opts,'valid')&&(numel(opts.valid)>0)
                plot(opts.results.ValidEpochLoss,'g','DisplayName','Valid');hold on;            
            end            
            plot(opts.results.TestEpochLoss,'r','DisplayName','Test');hold off;
            title('Loss per Epoch');legend('show')
            drawnow;
        end
        
    end
    
    parameters=opts.parameters;
    results=opts.results;
    save([fullfile(opts.output_dir,[opts.output_name,num2str(ep),'.mat'])],'net','parameters','results');     
    
    opts.parameters.current_ep=opts.parameters.current_ep+1;
    
end

opts.train=[];
opts.test=[];

if strcmp(net.layers{end}.type,'softmaxloss')
    if isfield(opts,'valid')&&(numel(opts.valid)>0)
        [min_err_valid,best_id]=min(opts.results.ValidEpochError);
         min_err=opts.results.TestEpochError(best_id);  
         disp(['Model validation error rate: ',num2str(min_err_valid)]);
    else
        [min_err,best_id]=min(opts.results.TestEpochError);
    end
    disp(['Model test error rate: ',num2str(min_err)]);
    best_net_source=[fullfile(opts.output_dir,[opts.output_name,num2str(best_id),'.mat'])];
    best_net_destination=[fullfile(opts.output_dir,['best_',opts.output_name,num2str(best_id),'.mat'])];
    copyfile(best_net_source,best_net_destination);
end

saveas(gcf,[fullfile(opts.output_dir,[opts.output_name,num2str(opts.n_epoch),'.pdf'])])



