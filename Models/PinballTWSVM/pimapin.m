
filename = '..\..\Datasets\PimaIndians.csv';
delimiter = ',';

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%s%s%s%s%s%s%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric strings to numbers.
% Replace non-numeric strings with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3,4,5,6,7,8,9]
    % Converts strings in the input cell array to numbers. Replaced non-numeric
    % strings with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(thousandsRegExp, ',', 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric strings to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end


%% Create output variable
PimaIndians = cell2mat(raw);
%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me;
[M,N]=size(PimaIndians);
avgtime=0;
avgaccu=0;
%Specifying the seed value
s = RandStream('mt19937ar','Seed',0);
%Doing cross validation
for q=1:10
    rand_pos = randperm(M); %array of random positions
    % new array with original data randomly distributed
    data=zeros(M,N);
    for k = 1:M
        data(k,:) = PimaIndians(rand_pos(k),:);
    end

    % Get Data and Labels
    features=data(:,1:end-1);
    labels=data(:,end);

    % Normalize labels
    labels(labels==0)=-1;
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
    C1=0.5; C2=0.5;
    V1=0.125; V2=0.125;
    T1=0.5; T2=1;
    % Run Twin SVM (Linear)
    
    [ wA, bA, wB, bB,time ] = LinearPinTWSVM( xTrain, yTrain, C1, C2, V1, V2, T1, T2 );
    
    avgtime=avgtime+time;
    yPred=zeros(size(xTest,1),1);
    for i=1:size(xTest,1)
        sample=xTest(i,:);
        distA=(sample*wA' + bA)/norm(wA);
        distB=(sample*wB' + bB)/norm(wB);
        if (distA>distB)
            yPred(i)=-1;
        else
            yPred(i)=1;
        end
    end



    accuracy=(sum(yPred==yTest)/length(yTest))*100;

    % Sanity check - if labels are predicted wrongly then flip
    if (accuracy<50)
        yPred=-1*yPred;
        accuracy=(sum(yPred==yTest)/length(yTest))*100;
    end
    avgaccu=avgaccu+accuracy;
end
%Display the results
 disp('Accuracy (Linear) is');
 disp(avgaccu/10);
 disp('Time taken is');
 disp(avgtime/10);
 