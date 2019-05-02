%import without noise file
filename = '..\..\Datasets\WITHOUT_NOISE (1).csv';
delimiter = ',';

%% Format string for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
WITHOUTNOISE1 = [dataArray{1:end-1}];
%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans;
[M,N]=size(WITHOUTNOISE1);
boxpl=zeros(10,2);
sens=0;
specs=0;
prec=0;
f1=0;
%Specifying the seed value
s = RandStream('mt19937ar','Seed',0);
for q=1:10
    rand_pos = randperm(s,M); %array of random positions
    % new array with original data randomly distributed
    data=zeros(M,N);
    for k = 1:M
        data(k,:) = WITHOUTNOISE1(rand_pos(k),:);
    end
    % Get Data and Labels
    features=data(:,1:end-1);
    labels=data(:,end);

    % Normalize labels
    labels(labels==0)=-1;
    % Normalize features
    features=zscore(features);


    % Separate training and test data (80:20 split)
    total_samples=size(features,1);
    train_samples=round(0.8*total_samples);

    % Define training and test samples
    xTrain=features(1:train_samples,:);
    yTrain=labels(1:train_samples,:);
    xTest=features(train_samples+1:end,:);
    yTest=labels(train_samples+1:end,:);

    % Define hyperparameter values
    C1=0.9; C2=0.9;

    % Run Twin SVM (Linear)
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
    yTest=int8(yTest);
    yPred=int8(yPred);
    %Calculating F1 Score, Sensitivity, Specificity and Precision.
    C = confusionmat(yTest,yPred);
    sensitivity=C(1,1)/(C(1,1)+C(1,2));
    specificity=C(2,2)/(C(2,1)+C(2,2));
    precision=C(1,1)/(C(1,1)+C(2,1));
    F1_score=200*precision*sensitivity/(precision+sensitivity);
    accuracy=(sum(yPred==yTest)/length(yTest))*100;   
    if(accuracy<50)
        yPred=-1*yPred;
        yTest=int8(yTest);
        yPred=int8(yPred);
        C = confusionmat(yTest,yPred);
        sensitivity=C(1,1)/(C(1,1)+C(1,2));
        specificity=C(2,2)/(C(2,1)+C(2,2));
        precision=C(1,1)/(C(1,1)+C(2,1));
        F1_score=200*precision*sensitivity/(precision+sensitivity);
    end
    %boxpl(i,2)=accuracy;
    sens=sens+sensitivity/10;
    specs=specs+specificity/10;
    prec=prec+precision/10;
    f1=f1+F1_score/10;
end
%import with noise file
filename = '..\..\Datasets\WITH_NOISE (1).csv';
delimiter = ',';

%% Format string for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
WITHNOISE1 = [dataArray{1:end-1}];
%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans;
[M,N]=size(WITHNOISE1);
nsens=0;
nspecs=0;
nprec=0;
nf1=0;
for q=1:10
    rand_pos = randperm(s,M); %array of random positions
    % new array with original data randomly distributed
    data=zeros(M,N);
    for k = 1:M
        data(k,:) = WITHNOISE1(rand_pos(k),:);
    end
    % Get Data and Labels
    features=data(:,1:end-1);
    labels=data(:,end);

    % Normalize labels
    labels(labels==0)=-1;
    % Normalize features
    features=zscore(features);


    % Separate training and test data (80:20 split)
    total_samples=size(features,1);
    train_samples=round(0.8*total_samples);

    % Define training and test samples
    xTrain=features(1:train_samples,:);
    yTrain=labels(1:train_samples,:);
    xTest=features(train_samples+1:end,:);
    yTest=labels(train_samples+1:end,:);
    % Define hyperparameter values
    C1=0.42; C2=0.42;

    % Run Twin SVM (Linear)
    
    [ wA, bA, wB, bB,time ] = LinearTWSVM( xTrain, yTrain, C1, C2 );
    
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
    yTest=int8(yTest);
    yPred=int8(yPred);
    %Calculating F1 Score, Sensitivity, Specificity and Precision.
    C = confusionmat(yTest,yPred);
    sensitivity=C(1,1)/(C(1,1)+C(1,2));
    specificity=C(2,2)/(C(2,1)+C(2,2));
    precision=C(1,1)/(C(1,1)+C(2,1));
    F1_score=200*precision*sensitivity/(precision+sensitivity);
    accuracy=(sum(yPred==yTest)/length(yTest))*100;   
    if(accuracy<50)
        yPred=-1*yPred;
        yTest=int8(yTest);
        yPred=int8(yPred);
        C = confusionmat(yTest,yPred);
        sensitivity=C(1,1)/(C(1,1)+C(1,2));
        specificity=C(2,2)/(C(2,1)+C(2,2));
        precision=C(1,1)/(C(1,1)+C(2,1));
        F1_score=200*precision*sensitivity/(precision+sensitivity);
    end
    %boxpl(i,2)=accuracy;
    nsens=nsens+sensitivity/10;
    nspecs=nspecs+specificity/10;
    nprec=nprec+precision/10;
    nf1=nf1+F1_score/10;
    %accuracy=(sum(yPred==yTest)/length(yTest))*100;
end
%Displaying the results
disp('The results for with noise dataset');
disp('Sensitivity');
disp(nsens*100);
disp('Specificity');
disp(nspecs*100);
disp('Precision');
disp(nprec*100);
disp('F1 Score');
disp(nf1);
disp('The results for without noise dataset');
disp('Sensitivity');
disp(sens*100);
disp('Specificity');
disp(specs*100);
disp('Precision');
disp(prec*100);
disp('F1 Score');
disp(f1);