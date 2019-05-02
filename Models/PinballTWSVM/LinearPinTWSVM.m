function [ wA, bA, wB, bB,time ] = LinearPinTWSVM( xTrain, yTrain, C1, C2, V1, V2, T1, T2 )
[N,D]=size(xTrain);

% Separate data of the two classes
A=xTrain(yTrain==1,:);
B=xTrain(yTrain==-1,:);

% Obtain Twin SVM hyperplanes
[ wA, bA,time1 ] = PinLTWSVM1( A, B, C1,V1,T1 );
[ wB, bB,time2 ] = PinLTWSVM2( A, B, C2,V2,T2 );
%Getting time
time=time1+time2;
end

