%% Import feature data and wind speed data for the day
data = xlsread('your_data.xlsx');
Features = data(1:18,:); %% Feature input: 75 days, 24 hours per day, one sample per hour, totaling 7524=1800 hours, 18 feature data
Wind_data = data(19,:); %% Actual output: wind speed data for 7524=1800 hours
%% Flatten the data into 4-D
LP_Features = double(reshape(Features,18,24,1,75)); %% Feature data format is 1824175, corresponding to 18 features for 24 hours across 75 days
LP_WindData = double(reshape(Wind_data,24,1,1,75)); %% Actual data format is 241175 , corresponding to 24 hours across 75 days
%% Format conversion to cell
NumDays = 75; %% Total number of days is 75
for i=1:NumDays
FeaturesData{1,i} = LP_Features(:,:,1,i);
end
for i=1:NumDays
RealData{1,i} = LP_WindData(:,:,1,i);
end
%% Data partitioning
XTrain = FeaturesData(:,1:73); %% Training input is the feature data from days 1-73
YTrain = RealData(:,2:74); %% Training output is the actual data from days 2-74
XTest = cell2mat(FeaturesData(: , 74)); %% Test input is the feature data of day 74
Ytest = cell2mat(RealData(: , 75)); %% Test output is the actual data of day 75
%% Define the hyperparameters directly
learning_rate = 0.01; %% Learning rate
KerlSize = 3; %% Kernel size
NumNeurons = 32; %% Number of neurons
%% Network construction
lgraph = [
sequenceInputLayer([18 24 1],"Name","sequence")
convolution2dLayer(KerlSize,3,"Name","conv","Padding","same")
batchNormalizationLayer("Name","batchnorm")
reluLayer("Name","relu")
maxPooling2dLayer([3 3],"Name","maxpool","Padding","same")
flattenLayer("Name","flatten")
lstmLayer(NumNeurons,"Name","lstm")
fullyConnectedLayer(24,"Name","fc")
regressionLayer("Name","regressionoutput")];
%% Set the training parameters
options = trainingOptions('sgdm', ...
'MaxEpochs',400, ...
'GradientThreshold',1,...
'ExecutionEnvironment','cpu',...
'InitialLearnRate',learning_rate,...
'LearnRateSchedule','none',...
'Shuffle','every-epoch',...
'SequenceLength',24,...
'MiniBatchSize',15,...
'Verbose',true);
%% Train the network
[net,info] = trainNetwork(XTrain,YTrain, lgraph, options);
%% Testing and evaluation
YPredicted = net.predict(XTest);
tsmvalue = YPredicted;
%% Calculate the error
error2 = YPredicted-Ytest;
[~,len]=size(Ytest);
SSE1=sum(error2.^2);
MAE1=sum(abs(error2))/len;
MSE1=error2*error2'/len;
RMSE1=MSE1^(1/2);
MAPE1=mean(abs(error2./mean(Ytest)));
r=corrcoef(Ytest,YPredicted);
R1=r(1,2);
R=MAPE1;
display(['Current batch MAPE:', num2str(R)]);