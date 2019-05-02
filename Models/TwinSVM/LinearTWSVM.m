function [ wA, bA, wB, bB ,time ] = LinearTWSVM( xTrain, yTrain, C1, C2 )

[N,D]=size(xTrain);


% Separate data of the two classes
A=xTrain(yTrain==1,:);
B=xTrain(yTrain==-1,:);

% Obtain Twin SVM hyperplanes

[ wA, bA ,time1 ] = LTWSVM1( A, B, C1 );
[ wB, bB, time2 ] = LTWSVM2( A, B, C2 );
%time
time=time1+time2;

end

