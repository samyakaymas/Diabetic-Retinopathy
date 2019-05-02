function [ accuracy ] = accusvm( xTrain,yTrain, xTest, yTest, C)
%Function to predict the accuracy
[ w,b,time ] = SVM( xTrain, yTrain, C );
yTest(yTest==-1)=0;
[n,m]=size(xTest);
yTest=int8(yTest);
yPred=(xTest*w'+b*ones(n,1))>0;
accuracy=(sum(yPred==yTest)/length(yTest))*100;
if(accuracy<50)
    accuracy=100-accuracy;
end
end