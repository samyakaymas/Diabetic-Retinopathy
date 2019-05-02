function [ bestAcc, C1_best, C2_best ] = TuneLinearTwinSVM( trainData, trainLabels )

% Initializations
C1_range=[-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10];
C1_range=2.^C1_range;
C2_range=[-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10];
C2_range=2.^C2_range;
% Separate validation set
[N, D]=size(trainData);
split_pt=round(0.8*N);
xTrain=trainData(1:split_pt,:);
yTrain=trainLabels(1:split_pt,:);
xTest=trainData(split_pt+1:end,:);
yTest=trainLabels(split_pt+1:end,:);
bestAcc=0;
C1_best=0;C2_best=0;



% Tune
for i=1:length(C1_range)
    C1=C1_range(i);
    for j=1:length(C2_range)
        C2=C2_range(j);
        
        % Train and test twin SVM
        [ wA, bA, wB, bB,time] = LinearTWSVM( xTrain, yTrain, C1, C2 );
        yPred=zeros(size(xTest,1),1);
        for i=1:size(xTest,1)
            sample=xTest(i,:);
            distA=(sample*wA + bA)/norm(wA);
            distB=(sample*wB + bB)/norm(wB);
            if (distA>distB)
                yPred(i)=-1;
            else
                yPred(i)=1;
            end
        end
        accuracy=(sum(yPred==yTest)/length(yTest))*100;
        if (accuracy<50)
            yPred=-1*yPred;
            accuracy=(sum(yPred==yTest)/length(yTest))*100;
        end
        if (accuracy>bestAcc)
            bestAcc=accuracy;
            C1_best=C1;
            C2_best=C2;
        end
    end
end


end

