function [ y,dzdy ] = pad_data_1d(  I,P,dzdy)
%PAD_DATA Summary of this function goes here
%   Detailed explanation goes here

    if(length(P)==1)
        P=ones(1,2)*P;
    end
    
        
    if isempty(dzdy)
        %%forward
        original_size=size(I);
        new_size=original_size;
        new_size(1)=new_size(1)+P(1)+P(2);
        y=zeros(new_size,'like',I);
        y(1+P(1):P(1)+original_size(1),:,:)=I;

    else
        %backward
        new_size=size(dzdy);
        original_size=new_size;
        original_size(1)=original_size(1)-P(1)-P(2);
        y=I(1+P(1):P(1)+original_size(1),:,:);
        dzdy=dzdy(1+P(1):P(1)+original_size(1),:,:);
        
    end

end

