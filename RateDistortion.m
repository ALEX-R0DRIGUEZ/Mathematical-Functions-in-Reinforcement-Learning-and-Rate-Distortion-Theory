LloydTrainMSE(1)
LloydTrainMSE(2)
LloydTrainMSE(3)

%calculation distortion
function D = distort(Q,T,M)
    D = 0;
    for i = 1:M
        D = D + (T(i)-Q(i))^2;
    end
    D = D/M;
end

%Lloyd algo for training data and MSE
function LloydTrainMSE(R)
    N = 2^R;
    M = 5000;
    
%Step 1
    Q = zeros(1,M);
    %training set
    T = normrnd(0,1,M,1);
    %initial codebook
    C = linspace(-1,1,N);
    %threshold
    e = 0.001;
    
    %calculates Q in relation to T and C
    for i = 1:M
        min = M/N;
        intClose = 0;
        for j = 1:N
            if abs(C(j)-T(i)) < min
                intClose = j;
                min = (C(j)-T(i))^2;
            end
        end
        Q(i) = C(intClose);
    end
    
    %calculate D_m
    D1 = distort(Q,T,M);
    
%Step 2
    thresholdCheck = 1;
    while thresholdCheck >= e
        bin = zeros(1,N);
        %partitions T into N bins using NNC
        for i = 1:M
            min = M/N;
            intClose = 0;
            for j = 1:N
                if abs(T(i)-C(j)) < min
                    intClose = j;
                    min = abs(C(j)-T(i));
                end
            end
            %creates bins
            bin(end+1,intClose) = T(i);
        end
        
        %generates C_m+1
        for i = 1:N
            empiricalAverage = nonzeros(bin(:,i));
            C(i) = sum(empiricalAverage)/length(empiricalAverage);
        end
        
        %updates Q
        for i = 1:M
            min = M/N;
            intClose = 0;
            for j = 1:N
                if abs(C(j)-T(i)) < min
                    intClose = j;
                    min = abs(C(j)-T(i));
                end
            end
            Q(i) = C(intClose);
        end
        
%Step 3
        %calculate D_m+1
        D2 = distort(Q,T,M);
        %end of step 3 condition to output C_m+1
        if abs((D1-D2)/D1) < e
            fprintf('R = %d bits/sample\n',R);
            disp('Q = ');
            disp(C);
            %calculate final distortion
            D = distort(Q,T,M);
            disp('D = ');
            disp(D);
            break;
        else
            %m=m+1 and go to step 2
            thresholdCheck = abs((D2-D1)/D2);
            D1 = D2;
        end
    end
end