function [ y,dzdy ] = pad_data(  I,P,dzdy)
%PAD_DATA Summary of this function goes here
%   Detailed explanation goes here

    if(length(P)==1)
        P=ones(1,4)*P;
    end
    
        
    if isempty(dzdy)
        %%forward
        original_size=size(I);
        new_size=original_size;
        new_size(1)=new_size(1)+P(1)+P(2);
        new_size(2)=new_size(2)+P(3)+P(4);
        y=zeros(new_size,'like',I);
        y(1+P(1):P(1)+original_size(1),1+P(3):P(3)+original_size(2),:,:)=I;

    else
        %backward
        new_size=size(dzdy);
        original_size=new_size;
        original_size(1)=original_size(1)-P(1)-P(2);
        original_size(2)=original_size(2)-P(3)-P(4);
        y=I(1+P(1):P(1)+original_size(1),1+P(3):P(3)+original_size(2),:,:);
        dzdy=dzdy(1+P(1):P(1)+original_size(1),1+P(3):P(3)+original_size(2),:,:);
        
    end

end

