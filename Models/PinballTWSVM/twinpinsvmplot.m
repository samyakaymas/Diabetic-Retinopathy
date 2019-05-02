%Getting accuracy on data with noise

%% Initialize variables.
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
WITHNOISE2 = [dataArray{1:end-1}];
%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans;
[M,N]=size(WITHNOISE2);
%Defining Hyperparameter
v=0.125;
t1=0.1;
t2=0.1;
C=[-4,-3,-2,-1,0,1,2];
C=2.^C;
for i=1:length(C)
    c=C(i);
    avgaccu=0;
    for ite=1:10
        rand_pos = randperm(M); %array of random positions
        % new array with original data randomly distributed
        data=zeros(M,N);
        for k = 1:M
            data(k,:) = WITHNOISE2(rand_pos(k),:);
        end

        features=data(:,1:end-1);
        labels=data(:,end);
        % Normalize labels
        features=zscore(features);
        labels(labels==0)=-1;
        % Normalize labels


        % Separate training and test data (80:20 split)
        total_samples=size(features,1);
        train_samples=round(0.8*total_samples);

        % Define training and test samples
        xTrain=features(1:train_samples,:);
        yTrain=labels(1:train_samples,:);
        xTest=features(train_samples+1:end,:);
        yTest=labels(train_samples+1:end,:);
        
        [k]=accutwinpinsvm(xTrain,yTrain,xTest,yTest,c,c,v,v,t1,t2);
        avgaccu=avgaccu+k;
    end
    accu(i)=avgaccu/10;
end


%getting accuracy on data without noise
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
C=[-4,-3,-2,-1,0,1,2];
C=2.^C;
for i=1:length(C)
    c=C(i);
    avgaccu=0;
    for ite=1:10
        rand_pos = randperm(M); %array of random positions
        % new array with original data randomly distributed
        data=zeros(M,N);
        for k = 1:M
            data(k,:) = WITHOUTNOISE1(rand_pos(k),:);
        end

        features=data(:,1:end-1);
        labels=data(:,end);
        % Normalize labels
        features=zscore(features);
        labels(labels==0)=-1;
        % Normalize labels


        % Separate training and test data (80:20 split)
        total_samples=size(features,1);
        train_samples=round(0.8*total_samples);

        % Define training and test samples
        xTrain=features(1:train_samples,:);
        yTrain=labels(1:train_samples,:);
        xTest=features(train_samples+1:end,:);
        yTest=labels(train_samples+1:end,:);
        
        [k]=accutwinpinsvm(xTrain,yTrain,xTest,yTest,c,c,v,v,t1,t2);
        avgaccu=avgaccu+k;
    end
    accu_without(i)=avgaccu/10;
end

%plotting the graph
figure
%Finding peak
[maxvalue, ind] = max(accu);
plot(C,accu,'r',C(ind),maxvalue,'or');
hold on 
%Finding peak
[maxvalue1, ind1] = max(accu_without);
plot(C,accu_without,'b',C(ind1),maxvalue1,'or');
legend('with noise',strcat(num2str(maxvalue),',',num2str(C(ind))),'without noise',strcat(num2str(maxvalue1),',',num2str(C(ind1))));
xlabel('Regularization Parameter C');
ylabel('Accuracy');
title('Accuracy vs Regularization Parameter');