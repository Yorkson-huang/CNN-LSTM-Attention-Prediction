% Kepler优化算法（KOA）
function [Sun_Score, Best_Pos, KOA_curve, bestPred,bestNet,bestInfo ] = KOA(SearchAgents_no, Tmax, ub, lb, dim)

%%  定义
Sun_Pos   = zeros(1, dim);  %% 包含迄今为止的最优解的向量，表示太阳
Sun_Score = inf;            %% 包含迄今为止的最优分数的标量

%%  控制参数
%%
Tc = 3;
M0 = 0.1;
lambda = 15;
%% 第1步：初始化过程
% 轨道离心率 (e)   
orbital = rand(1, SearchAgents_no);                      %% Eq.(4)

 %% 轨道周期 (T) 
T = abs(randn(1, SearchAgents_no));                      %% Eq.(5)
Positions = initialization(SearchAgents_no, dim, ub, lb);%% 初始化行星位置
t = 0; %% 函数评估计数器 
%%
%%---------------------评估-----------------------%%
for i = 1:SearchAgents_no
    %% 目标函数嵌套
    [PL_Fit(i),tsmvalue{i},tnet{i},tinfo{i}] = objectiveFunction(Positions(i,:)');    
    % 更新迄今为止的最优解
    if PL_Fit(i) < Sun_Score      %% 问题为最大化时，请将其更改为>
       Sun_Score = PL_Fit(i);     %% 更新迄今为止的最优分数
       Sun_Pos = Positions(i,:);  %% 更新迄今为止的最优解
       bestPred = tsmvalue{i} ;   %% 更新迄今为止的最准确预测结果
       bestNet = tnet{i};
       bestInfo  = tinfo{i};
    end
end

while t < Tmax            %% 终止条件  
 [Order] = sort(PL_Fit);  %% 对当前种群中的解的适应度值进行排序
 %% 函数评估t时的最差适应度值
 worstFitness = Order(SearchAgents_no);                  %% Eq.(11)
 M = M0 * (exp(-lambda * (t / Tmax)));                   %% Eq.(12)

 %% 计算表示太阳与第i个解之间的欧几里得距离R
 for i = 1:SearchAgents_no
    R(i) = 0;
    for j = 1:dim
       R(i) = R(i) + (Sun_Pos(j) - Positions(i, j))^2;   %% Eq.(7)
    end
    R(i) = sqrt(R(i));
 end
 %% 太阳和对象i在时间t的质量计算如下：
 for i = 1:SearchAgents_no
    sum = 0;
    for k = 1:SearchAgents_no
        sum = sum + (PL_Fit(k) - worstFitness);
    end
    MS(i) = rand * (Sun_Score - worstFitness) / (sum);   %% Eq.(8)
    m(i) = (PL_Fit(i) - worstFitness) / (sum);           %% Eq.(9)
 end
 
 %% 第2步：定义引力（F）
 % 计算太阳和第i个行星的引力，根据普遍的引力定律：
 for i = 1:SearchAgents_no
    Rnorm(i) = (R(i) - min(R)) / (max(R) - min(R));      %% 归一化的R（Eq.(24)）
    MSnorm(i) = (MS(i) - min(MS)) / (max(MS) - min(MS)); %% 归一化的MS
    Mnorm(i) = (m(i) - min(m)) / (max(m) - min(m));      %% 归一化的m
    Fg(i) = orbital(i) * M * ((MSnorm(i) * Mnorm(i)) / (Rnorm(i) * Rnorm(i) + eps)) + (rand); %% Eq.(6)
 end
% a1表示第i个解在时间t的椭圆轨道的半长轴，
for i = 1:SearchAgents_no
    a1(i) = rand * (T(i)^2 * (M * (MS(i) + m(i)) / (4 * pi * pi)))^(1/3); %% Eq.(23)
end

for i = 1:SearchAgents_no
% a2是逐渐从-1到-2的循环控制参数
a2 = -1 - 1 * (rem(t, Tmax / Tc) / (Tmax / Tc)); %% Eq.(29)

% ξ是从1到-2的线性减少因子
n = (a2 - 1) * rand + 1;    %% Eq.(28)
a = randi(SearchAgents_no); %% 随机选择的解的索引
b = randi(SearchAgents_no); %% 随机选择的解的索引
rd = rand(1, dim);          %% 按照正态分布生成的向量
r = rand;                   %% r1是[0,1]范围内的随机数

%% 随机分配的二进制向量
U1 = rd < r;                %% Eq.(21)
O_P = Positions(i, :);      %% 存储第i个解的当前位置

%% 第6步：更新与太阳的距离（第3、4、5在后面）
if rand < rand
    % h是一个自适应因子，用于控制时间t时太阳与当前行星之间的距离
    h = (1 / (exp(n * randn))); %% Eq.(27)
    % 基于三个解的平均向量：当前解、迄今为止的最优解和随机选择的解
    Xm = (Positions(b, :) + Sun_Pos + Positions(i, :)) / 3.0;
    Positions(i, :) = Positions(i, :) .* U1 + (Xm + h .* (Xm - Positions(a, :))) .* (1 - U1); %% Eq.(26)
else
%% 第3步：计算对象的速度
    % 一个标志，用于相反或离开当前行星的搜索方向
     if rand < 0.5 %% Eq.(18)
       f = 1;
     else
       f = -1;
     end
     L = (M * (MS(i) + m(i)) * abs((2 / (R(i) + eps)) - (1 / (a1(i) + eps))))^(0.5); %% Eq.(15)
     U = rd > rand(1, dim); %% 一个二进制向量
     if Rnorm(i) < 0.5 %% Eq.(13)
        M = (rand .* (1 - r) + r); %% Eq.(16)
        l = L * M * U; %% Eq.(14)
        Mv = (rand * (1 - rd) + rd); %% Eq.(20)
        l1 = L .* Mv .* (1 - U);%% Eq.(19)
        V(i, :) = l .* (2 * rand * Positions(i, :) - Positions(a, :)) + l1 .* (Positions(b, :) - Positions(a, :)) + (1 - Rnorm(i)) * f * U1 .* rand(1, dim) .* (ub - lb); %% Eq.(13a)
     else
        U2 = rand > rand; %% Eq. (22) 
        V(i, :) = rand .* L .* (Positions(a, :) - Positions(i, :)) + (1 - Rnorm(i)) * f * U2 * rand(1, dim) .* (rand * ub - lb);  %% Eq.(13b)
     end %% 结束IF
     
%% 第4步：逃离局部最优
     % 更新标志f以相反或离开当前行星的搜索方向
     if rand < 0.5 %% Eq.(18)
        f = 1;
     else
        f = -1;
     end
%% 第5步
     Positions(i, :) = ((Positions(i, :) + V(i, :) .* f) + (Fg(i) + abs(randn)) * U .* (Sun_Pos - Positions(i, :))); %% Eq.(25)
end %% 结束IF

%% 返回超出搜索空间边界的搜索个体
if rand < rand
   for j = 1:size(Positions, 2)
      if  Positions(i, j) > ub(j)
          Positions(i, j) = lb(j) + rand * (ub(j) - lb(j));
      elseif  Positions(i, j) < lb(j)
          Positions(i, j) = lb(j) + rand * (ub(j) - lb(j));
      end %% 结束IF
   end   %% 结束For
else
   Positions(i, :) = min(max(Positions(i, :), lb), ub);
end %% 结束IF
%% 目标函数嵌套
% 为每个搜索个体计算目标函数
[PL_Fit1,tsmvalue1,tnet1,tinfo1] = objectiveFunction(Positions(i,:)');

%% 第7步：精英主义, Eq.(30)
if PL_Fit1 < PL_Fit(i)             %% 问题为最大化时，请将其更改为>
    PL_Fit(i)   = PL_Fit1; 
    tsmvalue{i} = tsmvalue1;
    tnet{i} = tnet1;
    tinfo{i} = tinfo1;
    
    % 更新迄今为止的最优解
    if PL_Fit(i) < Sun_Score       %% 问题为最大化时，请将其更改为>
        Sun_Score = PL_Fit(i);     %% 更新迄今为止的最优分数
        Sun_Pos = Positions(i, :); %% 更新迄今为止的最优解
        bestPred = tsmvalue{i};    %% 更新迄今为止的最准确预测结果
        bestNet = tnet{i};
        bestInfo  = tinfo{i};
    end
    else
    Positions(i, :) = O_P;
end %% 结束IF

t = t + 1;  %% 增加当前函数评估
if t > Tmax %% 检查终止条件
    break;
end % 结束IF

t = t + 1;  %% 增加当前函数评估
if t > Tmax %% 检查终止条件
    break;
end %% 结束IF

end %% 结束For i
end %% 结束While 

KOA_curve=sortrows(PL_Fit', -1);      % 适应度从高到低排序故为-1，绘制适应度曲线

%% 最终的参数优化结果
% 除了学习率其它位置均为整数
for i = 1:size(Sun_Pos,2)
    if i ==1
        Best_Pos(i) =  Sun_Pos(i);
    else
        Best_Pos(i) =  round(Sun_Pos(i));
    end  
end
end %% 结束KOA函数

%% 函数功能：初始化行星位置
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    % 随机生成初始位置
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end
