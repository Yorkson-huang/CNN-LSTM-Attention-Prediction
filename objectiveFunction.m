function [R,tsmvalue,net,info] = objectiveFunction(x)
%% 导入特征数据、当天的风速数据
data = xlsread('特征序列及实际值.xlsx');
Features   = data(1:18,:);                             %% 特征输入  ：75天，每天24小时，每小时一个采样点，共计75*24=1800小时，18个特征数据
Wind_data  = data(19,:);                               %% 实际值输出：75天，每天24小时，每小时一个采样点，共计75*24=1800小时的风速数据

%%  数据平铺为4-D
LP_Features =  double(reshape(Features,18,24,1,75));   %% 特征数据格式为18*24*1*75，分别对应18特征24小时，75天
LP_WindData  = double(reshape(Wind_data,24,1,1,75));   %% 实际数据格式为24*1*1*75 ，分别对应24小时，75天

%% 格式转换为cell
NumDays  = 75;                                         %% 数据总天数为 75天
for i=1:NumDays
    FeaturesData{1,i} = LP_Features(:,:,1,i);
end

for i=1:NumDays
    RealData{1,i} = LP_WindData(:,:,1,i);
end

%% 划分数据
XTrain = FeaturesData(:,1:73);                         %% 训练集输入为 1-73   天的特征
YTrain = RealData(:,2:74);                             %% 训练集输出为 2-74天 的实际值                

XTest  = cell2mat(FeaturesData(: , 74));               %% 测试集输入第  74    天的特征
Ytest  = cell2mat(RealData(: , 75));                   %% 测试集输出为第 75天 的实际值

%% 将优化目标参数传进来的值 转换为需要的超参数
learning_rate = x(1);            %% 学习率
KerlSize = round(x(2));          %% 卷积核大小
NumNeurons = round(x(3));        %% 神经元个数

%% 网络搭建
lgraph = [
    sequenceInputLayer([18 24 1],"Name","sequence")
    convolution2dLayer(KerlSize,3,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    maxPooling2dLayer([3 3],"Name","maxpool","Padding","same")
    flattenLayer("Name","flatten")
    lstmLayer(NumNeurons,"Name","lstm")
    selfAttentionLayer(1,24,"Name","selfattention")
    fullyConnectedLayer(24,"Name","fc")
    regressionLayer("Name","regressionoutput")];


%% 设置训练参数
options = trainingOptions('sgdm', ...   % sgdm 梯度下降算法
    'MaxEpochs',400, ...                % 最大训练次数 400
    'GradientThreshold',1,...           % 渐变的正阈值 1
    'ExecutionEnvironment','cpu',...    % 网络的执行环境 cpu
    'InitialLearnRate',learning_rate,...% 初始学习率 0.01
    'LearnRateSchedule','none',...      % 训练期间降低整体学习率的方法 不降低
    'Shuffle','every-epoch',...         % 每次训练打乱数据集
    'SequenceLength',24,...             % 序列长度 24
    'MiniBatchSize',15,...              % 训练批次大小 每次训练样本个数15
    'Verbose',true);                    % 有关训练进度的信息不打印到命令窗口中

%% 训练网络
[net,info] = trainNetwork(XTrain,YTrain, lgraph, options);

%% 测试与评估
YPredicted = net.predict(XTest);                       
tsmvalue = YPredicted;

%% 计算误差
% 过程
error2 = YPredicted-Ytest;            % 测试值和真实值的误差  
[~,len]=size(Ytest);                  % len获取测试样本个数，数值等于testNum，用于求各指标平均值
SSE1=sum(error2.^2);                  % 误差平方和
MAE1=sum(abs(error2))/len;            % 平均绝对误差
MSE1=error2*error2'/len;              % 均方误差
RMSE1=MSE1^(1/2);                     % 均方根误差
MAPE1=mean(abs(error2./mean(Ytest))); % 平均百分比误差
r=corrcoef(Ytest,YPredicted);         % corrcoef计算相关系数矩阵，包括自相关和互相关系数
R1=r(1,2); 
R=MAPE1;
display(['本批次MAPE:', num2str(R)]);
end

