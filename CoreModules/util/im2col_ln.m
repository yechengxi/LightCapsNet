function [ slices,idx0 ] = im2col_ln( I, K, S )

    [Hin,Win,N,B]=size(I);
    if length(K)==1
        K(2)=K(1);
    end
    if length(S)==1
        S(2)=S(1);
    end
    Hout = ceil((Hin-K(1)+1)/S(1));
    Wout = ceil((Win-K(2)+1)/S(2));
    
    
    %%%index the entries in a sliding window
    
    idx1= [1:K(1)]';
    idx1=idx1(:,ones(K(2),1));
    
    idx2=[0:K(2)-1]*Hin;
    idx2=idx2(ones(K(1),1),:);
    
    idx1=idx1+idx2;
    idx1=idx1(:);
    
    %%%index the window positions in the input image.
    
    idx3=[0:Hout-1]'*S(1);
    idx3=idx3(:,ones(Wout,1));
    
    idx4=[0:Wout-1]*Hin*S(2);
    idx4=idx4(ones(Hout,1),:);
    
    idx3=idx3+idx4;
    idx3=idx3(:)';
    
    %%%%this will give us the index of all the sliding window elements in
    %%%%an image.
    idx0=(idx1+idx3);
    
    %%%index the batches and channels.
    idx5=[0:N-1]'*Hin*Win;
    idx5=idx5(:,ones(B,1));
    
    idx6=[0:B-1]*Hin*Win*N;
    idx6=idx6(ones(N,1),:);
    
    idx5=(idx5+idx6);
    
    %%%indexes of all elements
    idx5=permute(idx5,[3,4,1,2]);
    if isa(I,'gpuArray')
        idx0=gpuArray(idx0);
        idx5=gpuArray(idx5);
    end
    idx0=(idx0+idx5); %better calculated on gpu
    
    
    %%%pooling computations
    slices=I(idx0);

end