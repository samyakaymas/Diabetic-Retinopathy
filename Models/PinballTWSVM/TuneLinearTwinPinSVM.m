function [ bestAcc,C_best, V_best,T1_best,T2_best ] = TuneLinearTwinPinSVM( trainData, trainLabels )


% Initializations

C_range=[-3,-2,-1,0,1,2,3,4,5,6,7,8];
C_range=2.^C_range;
V_range=[-3,-2,-1,0,1,2,3,4,5,6,7,8];
V_range=2.^V_range;
T_range=[0.1,0.2,0.5,1,0.05];

% Separate validation set
[N, D]=size(trainData);
split_pt=round(0.8*N);
xTrain=trainData(1:split_pt,:);

yTrain=trainLabels(1:split_pt,:);
xTest=trainData(split_pt+1:end,:);
yTest=trainLabels(split_pt+1:end,:);
bestAcc=0;
C_best=0;V_best=0;T1_best=0;T2_best=0;



% Tune
for i=1:length(C_range)
    C=C_range(i);
    for j=1:length(V_range)
        V=V_range(j);
        for k=1:length(T_range)
            T1=T_range(k);
            for k1=1:length(T_range)
            T2=T_range(k1);
            % Train and test twin SVM
            [ wA,bA,wB,bB,time ] = LinearPinTWSVM( xTrain, yTrain, C, C,V,V,T1,T2 );
            yPred=zeros(size(xTest,1),1);
            for i1=1:size(xTest,1)
                sample=xTest(i1,:);
                distA=(sample*wA' + bA)/norm(wA);
                distB=(sample*wB' + bB)/norm(wB);
                if (distA>distB)
                    yPred(i1)=-1;
                else
                    yPred(i1)=1;
                end
            end
            accuracy=(sum(yPred==yTest)/length(yTest))*100;

% Sanity check - if labels are predicted wrongly then flip
            if (accuracy<50)
                yPred=-1*yPred;
                accuracy=(sum(yPred==yTest)/length(yTest))*100;
            end
        
            if (accuracy>bestAcc)
                bestAcc=accuracy;
                C_best=C;
                V_best=V;
                T1_best=T1;
                T2_best=T2;
            end
            end
        end
    end
end
end

