% Kepler�Ż��㷨��KOA��
function [Sun_Score, Best_Pos, KOA_curve, bestPred,bestNet,bestInfo ] = KOA(SearchAgents_no, Tmax, ub, lb, dim)

%%  ����
Sun_Pos   = zeros(1, dim);  %% ��������Ϊֹ�����Ž����������ʾ̫��
Sun_Score = inf;            %% ��������Ϊֹ�����ŷ����ı���

%%  ���Ʋ���
%%
Tc = 3;
M0 = 0.1;
lambda = 15;
%% ��1������ʼ������
% ��������� (e)   
orbital = rand(1, SearchAgents_no);                      %% Eq.(4)

 %% ������� (T) 
T = abs(randn(1, SearchAgents_no));                      %% Eq.(5)
Positions = initialization(SearchAgents_no, dim, ub, lb);%% ��ʼ������λ��
t = 0; %% �������������� 
%%
%%---------------------����-----------------------%%
for i = 1:SearchAgents_no
    %% Ŀ�꺯��Ƕ��
    [PL_Fit(i),tsmvalue{i},tnet{i},tinfo{i}] = objectiveFunction(Positions(i,:)');    
    % ��������Ϊֹ�����Ž�
    if PL_Fit(i) < Sun_Score      %% ����Ϊ���ʱ���뽫�����Ϊ>
       Sun_Score = PL_Fit(i);     %% ��������Ϊֹ�����ŷ���
       Sun_Pos = Positions(i,:);  %% ��������Ϊֹ�����Ž�
       bestPred = tsmvalue{i} ;   %% ��������Ϊֹ����׼ȷԤ����
       bestNet = tnet{i};
       bestInfo  = tinfo{i};
    end
end

while t < Tmax            %% ��ֹ����  
 [Order] = sort(PL_Fit);  %% �Ե�ǰ��Ⱥ�еĽ����Ӧ��ֵ��������
 %% ��������tʱ�������Ӧ��ֵ
 worstFitness = Order(SearchAgents_no);                  %% Eq.(11)
 M = M0 * (exp(-lambda * (t / Tmax)));                   %% Eq.(12)

 %% �����ʾ̫�����i����֮���ŷ����þ���R
 for i = 1:SearchAgents_no
    R(i) = 0;
    for j = 1:dim
       R(i) = R(i) + (Sun_Pos(j) - Positions(i, j))^2;   %% Eq.(7)
    end
    R(i) = sqrt(R(i));
 end
 %% ̫���Ͷ���i��ʱ��t�������������£�
 for i = 1:SearchAgents_no
    sum = 0;
    for k = 1:SearchAgents_no
        sum = sum + (PL_Fit(k) - worstFitness);
    end
    MS(i) = rand * (Sun_Score - worstFitness) / (sum);   %% Eq.(8)
    m(i) = (PL_Fit(i) - worstFitness) / (sum);           %% Eq.(9)
 end
 
 %% ��2��������������F��
 % ����̫���͵�i�����ǵ������������ձ���������ɣ�
 for i = 1:SearchAgents_no
    Rnorm(i) = (R(i) - min(R)) / (max(R) - min(R));      %% ��һ����R��Eq.(24)��
    MSnorm(i) = (MS(i) - min(MS)) / (max(MS) - min(MS)); %% ��һ����MS
    Mnorm(i) = (m(i) - min(m)) / (max(m) - min(m));      %% ��һ����m
    Fg(i) = orbital(i) * M * ((MSnorm(i) * Mnorm(i)) / (Rnorm(i) * Rnorm(i) + eps)) + (rand); %% Eq.(6)
 end
% a1��ʾ��i������ʱ��t����Բ����İ볤�ᣬ
for i = 1:SearchAgents_no
    a1(i) = rand * (T(i)^2 * (M * (MS(i) + m(i)) / (4 * pi * pi)))^(1/3); %% Eq.(23)
end

for i = 1:SearchAgents_no
% a2���𽥴�-1��-2��ѭ�����Ʋ���
a2 = -1 - 1 * (rem(t, Tmax / Tc) / (Tmax / Tc)); %% Eq.(29)

% ���Ǵ�1��-2�����Լ�������
n = (a2 - 1) * rand + 1;    %% Eq.(28)
a = randi(SearchAgents_no); %% ���ѡ��Ľ������
b = randi(SearchAgents_no); %% ���ѡ��Ľ������
rd = rand(1, dim);          %% ������̬�ֲ����ɵ�����
r = rand;                   %% r1��[0,1]��Χ�ڵ������

%% �������Ķ���������
U1 = rd < r;                %% Eq.(21)
O_P = Positions(i, :);      %% �洢��i����ĵ�ǰλ��

%% ��6����������̫���ľ��루��3��4��5�ں��棩
if rand < rand
    % h��һ������Ӧ���ӣ����ڿ���ʱ��tʱ̫���뵱ǰ����֮��ľ���
    h = (1 / (exp(n * randn))); %% Eq.(27)
    % �����������ƽ����������ǰ�⡢����Ϊֹ�����Ž�����ѡ��Ľ�
    Xm = (Positions(b, :) + Sun_Pos + Positions(i, :)) / 3.0;
    Positions(i, :) = Positions(i, :) .* U1 + (Xm + h .* (Xm - Positions(a, :))) .* (1 - U1); %% Eq.(26)
else
%% ��3�������������ٶ�
    % һ����־�������෴���뿪��ǰ���ǵ���������
     if rand < 0.5 %% Eq.(18)
       f = 1;
     else
       f = -1;
     end
     L = (M * (MS(i) + m(i)) * abs((2 / (R(i) + eps)) - (1 / (a1(i) + eps))))^(0.5); %% Eq.(15)
     U = rd > rand(1, dim); %% һ������������
     if Rnorm(i) < 0.5 %% Eq.(13)
        M = (rand .* (1 - r) + r); %% Eq.(16)
        l = L * M * U; %% Eq.(14)
        Mv = (rand * (1 - rd) + rd); %% Eq.(20)
        l1 = L .* Mv .* (1 - U);%% Eq.(19)
        V(i, :) = l .* (2 * rand * Positions(i, :) - Positions(a, :)) + l1 .* (Positions(b, :) - Positions(a, :)) + (1 - Rnorm(i)) * f * U1 .* rand(1, dim) .* (ub - lb); %% Eq.(13a)
     else
        U2 = rand > rand; %% Eq. (22) 
        V(i, :) = rand .* L .* (Positions(a, :) - Positions(i, :)) + (1 - Rnorm(i)) * f * U2 * rand(1, dim) .* (rand * ub - lb);  %% Eq.(13b)
     end %% ����IF
     
%% ��4��������ֲ�����
     % ���±�־f���෴���뿪��ǰ���ǵ���������
     if rand < 0.5 %% Eq.(18)
        f = 1;
     else
        f = -1;
     end
%% ��5��
     Positions(i, :) = ((Positions(i, :) + V(i, :) .* f) + (Fg(i) + abs(randn)) * U .* (Sun_Pos - Positions(i, :))); %% Eq.(25)
end %% ����IF

%% ���س��������ռ�߽����������
if rand < rand
   for j = 1:size(Positions, 2)
      if  Positions(i, j) > ub(j)
          Positions(i, j) = lb(j) + rand * (ub(j) - lb(j));
      elseif  Positions(i, j) < lb(j)
          Positions(i, j) = lb(j) + rand * (ub(j) - lb(j));
      end %% ����IF
   end   %% ����For
else
   Positions(i, :) = min(max(Positions(i, :), lb), ub);
end %% ����IF
%% Ŀ�꺯��Ƕ��
% Ϊÿ�������������Ŀ�꺯��
[PL_Fit1,tsmvalue1,tnet1,tinfo1] = objectiveFunction(Positions(i,:)');

%% ��7������Ӣ����, Eq.(30)
if PL_Fit1 < PL_Fit(i)             %% ����Ϊ���ʱ���뽫�����Ϊ>
    PL_Fit(i)   = PL_Fit1; 
    tsmvalue{i} = tsmvalue1;
    tnet{i} = tnet1;
    tinfo{i} = tinfo1;
    
    % ��������Ϊֹ�����Ž�
    if PL_Fit(i) < Sun_Score       %% ����Ϊ���ʱ���뽫�����Ϊ>
        Sun_Score = PL_Fit(i);     %% ��������Ϊֹ�����ŷ���
        Sun_Pos = Positions(i, :); %% ��������Ϊֹ�����Ž�
        bestPred = tsmvalue{i};    %% ��������Ϊֹ����׼ȷԤ����
        bestNet = tnet{i};
        bestInfo  = tinfo{i};
    end
    else
    Positions(i, :) = O_P;
end %% ����IF

t = t + 1;  %% ���ӵ�ǰ��������
if t > Tmax %% �����ֹ����
    break;
end % ����IF

t = t + 1;  %% ���ӵ�ǰ��������
if t > Tmax %% �����ֹ����
    break;
end %% ����IF

end %% ����For i
end %% ����While 

KOA_curve=sortrows(PL_Fit', -1);      % ��Ӧ�ȴӸߵ��������Ϊ-1��������Ӧ������

%% ���յĲ����Ż����
% ����ѧϰ������λ�þ�Ϊ����
for i = 1:size(Sun_Pos,2)
    if i ==1
        Best_Pos(i) =  Sun_Pos(i);
    else
        Best_Pos(i) =  round(Sun_Pos(i));
    end  
end
end %% ����KOA����

%% �������ܣ���ʼ������λ��
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    % ������ɳ�ʼλ��
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end
