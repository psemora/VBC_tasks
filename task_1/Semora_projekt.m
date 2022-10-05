clear all; clc
rng default % For reproducibility
dj = @dejong5fcn; %-32,-32,0.998
rf = @rastriginsfcn; %0,0,0
rbf = @rosenbrock; %1,1,0
bounds_dj= [-65.536 65.536; -65.536 65.536];
bounds_rf= [-5.12 5.12; -5.12 5.12];
bounds_rbf= [-10 10; -10 10];
 
%%%USER OPTIONS%%%
RUNs = 1000;
D = 2; %dimension of fun
fun =rf; %function (dj / rf / rbf)
bounds_fun = bounds_rf; %bounds for plot function
%%%%%%%%%%%%%%%%%%%%

x0 = 20*ones(1, D); %Coordinates of init point for Simulated annealing
result = zeros(RUNs,6);

%%%%Genetic Algorithm
options = optimoptions('ga',...
                       'PopulationSize',100, ...
                       'CreationFcn','gacreationuniform',...
                       'SelectionFcn','selectionroulette', ...
                       'FitnessLimit', 0.0001);
                       
                           
%Possible options for GA:    
%'PopulationSize',randi(200),... 
%'CreationFcn','gacreationuniform',...                   
%'CreationFcn','gacreationlinearfeasible',...                   
%'SelectionFcn' ,@selectiontournament
%'MutationFcn', {@mutationgaussian, 10, 10},...
%'MutationFcn',{@mutationuniform, 0.5},...
%'MutationFcn',{@mutationuniform, 100},...
%'CrossoverFcn', {@crossoverintermediate, 10},...
%'CrossoverFcn',{@crossoverheuristic,10},...
%'MaxStallGenerations', 100,...
%'MaxGenerations',100,....
%'FunctionTolerance', 1e-02);
%'PlotFcn',{@gaplotbestf,@gaplotstopping, @gaplotdistance,@gaplotbestindiv},...


for i = 1:RUNs 
    t = cputime;
    [x,f,f1,o,population,scores] = ga(fun, D, options);
    e = cputime-t;
    result(i,1) = x(1); 
    result(i,2) = x(2);
    result(i,3) = f; 
    result(i,5) = o.funccount;
    result(i,4) = o.generations; 
    result(i,6) = e; 
end

% %%Simulated Annealing
% for i = 1:RUNs
%     t = cputime;
%     [x,f,fl,o]=simulannealbnd(fun,x0);
%     e = cputime-t;
%     result(i,1) = x(1); 
%     result(i,2) = x(2); 
%     result(i,3) = f; 
%     result(i,4) = o.funccount;  
%     result(i,5) = e; 
% end

% COMPUTING MAX, MIN, MEAN, MEDIAN, MODE, STD
for i = 1:6
    result(RUNs+1,i) = max(result(1:RUNs,i)); %max
    result(RUNs+2,i) = min(result(1:RUNs,i));%min
    result(RUNs+3,i) = mean(result(1:RUNs,i)); %průměr
    result(RUNs+4,i) = median(result(1:RUNs,i));  %median
    result(RUNs+5,i) = mode(result(1:RUNs,i)); %modus
    result(RUNs+6,i) = std(result(1:RUNs,i)); %směrodatná odchylka
end

  
    
% PLOT FUNCTION WITH THE BEST MINIMUM (only for 2D):
if D == 2
    plotobjective(fun,bounds_fun) 
end
[f1 f2] = find(result==result(RUNs+2,3));
x1_min = result(f1(1),f2(1)-2);
x2_min = result(f1(1),f2(1)-1);
plot3(x1_min,x2_min,result(RUNs+2,3),'.r', 'MarkerSize',30);
title('Function')
xlabel('x1')
ylabel('x2')
zlabel('y')


%Rosenbrock function
function scores = rosenbrock(xin)
    scores = zeros(size(xin,1),1);
    for i = 1:size(xin,1)
        p = xin(i,:);
        sum = 0;
        for j = 1:length(p)-1
            sum = sum +100*((p(j+1)-p(j)^2)^2+(p(j)-1)^2);
        end
        scores(i) = sum;
    end
end
