function [ bestAcc, C_best ] = TuneLinearSVM( features, labels )
%Splitting the dataset
total_samples=size(features,1);
train_samples=round(0.8*total_samples);
xTest=features(train_samples+1:end,:);
yTest=labels(train_samples+1:end,:);
yTest(yTest==-1)=0;

%Initialising the hyperparameters
C_range=[-4,-5,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10];
C_range=2.^C_range;
bestAcc=0;
%Tune
for i=1:length(C_range)
    C=C_range(i);
    [w,b,time]=SVM(features, labels, C );
    [n,m]=size(xTest);
    yPred=(xTest*w'+b*ones(n,1))>0;
    accuracy=(sum(yPred==yTest)/length(yTest))*100;   
    if(accuracy<50)
        accuracy=100-accuracy;
    end
    if (accuracy>bestAcc)
        bestAcc=accuracy;
        C_best=C;
    end
end
end