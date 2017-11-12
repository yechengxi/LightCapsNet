function [ slices,idx0 ] = unroll_ln( I, K, S )

    [Din,N,B]=size(I);
    Dout = ceil((Din-K+1)/S);
    
    %%%index the entries in a sliding window
    
    idx1= [1:K(1)]';
    
    %%%index the window positions in the input signal.
    
    idx3=[0:Dout-1]'*S;
    idx3=idx3(:)';
    
    %%%%this will give us the index of all the sliding window elements in
    %%%%a 1d signal.
    idx0=(idx1+idx3);
    
    %%%index the batches and channels.
    idx5=[0:N-1]'*Din;
    idx5=idx5(:,ones(B,1));
    
    idx6=[0:B-1]*Din*N;
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