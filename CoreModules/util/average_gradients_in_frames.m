function [ Accum_Grad ] = average_gradients_in_frames( Frames )
%AVERAGE_GRADIENTS_IN_FRAMES Summary of this function goes here
%   Detailed explanation goes here
n_frames=length(Frames);
Accum_Grad=[];
for layer=1:length(Frames{1})
    if isfield(Frames{1}(layer),'dzdw')&& ~isempty(Frames{1}(layer).dzdw)
        Accum_Grad(layer).dzdw=0;
        for f=1:n_frames
            Accum_Grad(layer).dzdw=Accum_Grad(layer).dzdw+Frames{f}(layer).dzdw;
        end 
        Accum_Grad(layer).dzdw=Accum_Grad(layer).dzdw./n_frames;
    end

    if isfield(Frames{1}(layer),'dzdb')
        if ~isempty(Frames{1}(layer).dzdb)
            Accum_Grad(layer).dzdb=0;
            for f=1:n_frames
                Accum_Grad(layer).dzdb=Accum_Grad(layer).dzdb+Frames{f}(layer).dzdb;
            end 
            Accum_Grad(layer).dzdb=Accum_Grad(layer).dzdb./n_frames;
        else
            Accum_Grad(layer).dzdb=[];
        end
    end
end
    

end

