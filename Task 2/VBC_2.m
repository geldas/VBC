clear;clc;

ras_fun = @rastriginsfcn
dj_fun = @dejong5fcn
ros_fun = @rosenbrock

RUNs = 5; % number of runs
result_table = zeros(RUNs,6); % matrix with results

bound_ras1= -5.12; % function bounds
bound_ras2 = 5.12;
bound_dj1 = -65.536;
bound_dj2 = 65.536;
bound_ros1 = -2.048;
bound_ros2 = 2.048;

f = dj_fun; % current function
D = 2; % dimension (number of variables)
bound1 = bound_dj1; %bounds for current function
bound2 = bound_dj2;


%==========================================================================
%genetic algorithm

n_par = 6;
g_options = optimoptions('ga', ...
    'SelectionFcn','selectiontournament', ...
    'MaxGenerations',10000, ...
    'PopulationSize',300, ...
    'CreationFcn','gacreationuniform', ...
    'CrossoverFcn','crossoverlaplace', ...
    'EliteCount',0.08*300);

for i=1:RUNs
    t1 = cputime;
    [x,fval,exitflag,output] = ga(f,D,g_options);
    t2 = cputime;
    t = t2-t1;

    result_table(i,1) = x(1);
    result_table(i,2) = x(2);
    result_table(i,3) = fval;
    result_table(i,4) = output.funccount;
    result_table(i,5) = output.generations;
    result_table(i,6) = t;
end

% %==========================================================================
% % simulated annealing
% 
% n_par = 5;
% for i=1:RUNs
%     start_point = (bound2-bound1)*rand(1,D)+bound1;
%     t1 = cputime;
%     [x,fval,exitflag,output] = simulannealbnd(f,start_point);
%     t2 = cputime;
%     t = t2-t1;
% 
%     result_table(i,1) = x(1);
%     result_table(i,2) = x(2);
%     result_table(i,3) = fval;
%     result_table(i,4) = output.funccount;
%     result_table(i,5) = t;
% end
% 
% %==========================================================================
% %statistics
% 
% stats_n = 4;
% result_stats = zeros(stats_n,n_par);
% 
% for i=1:n_par
%     result_stats(1,i) = mean(result_table(1:RUNs,i));
%     result_stats(2,i) = min(result_table(1:RUNs,i));
%     result_stats(3,i) = max(result_table(1:RUNs,i));
%     result_stats(4,i) = median(result_table(1:RUNs,i));
% end
% 
% minpos = find(result_table(:,3) == result_stats(2,3));
% best = result_table(minpos,:);
% 
% 
% x_axis = 1:size(result_table(:,3),1);
% % 
% figure(1)
% scatter(x_axis,result_table(:,3),10,'filled')
% title('fval of RUN')
% xlabel('RUN')
% ylabel('fval')
% yline(result_stats(1,3),'-r','Mean');
% % yline([result_stats(2,3) result_stats(3,3)],'--g',{'Min','Max'});
% 
% % for genetic algorithm - column = 6
% % for simulatid annealing - column = 5
% figure(2)
% scatter(x_axis,result_table(:,5),10,'filled')
% title('Time of RUN')
% xlabel('RUN')
% ylabel('time')
% yline(result_stats(1,5),'-r','Mean')
% % yline([result_stats(2,5) result_stats(3,5)],'--g',{'Min','Max'});
% 
% figure(3)
% scatter(x_axis,result_table(:,4),10,'filled')
% title('#Eval of RUN')
% xlabel('RUN')
% ylabel('#Eval')
% yline(result_stats(1,4),'-r','Mean')
% % yline([result_stats(2,4) result_stats(3,4)],'--g',{'Min','Max'});
% 
% figure(4)
% plotobjective(f,[bound1, bound2; bound1, bound2])
% 
% plot3(best(1,1),best(1,2),best(1,3),Marker=".",Color="r",MarkerSize=20)
% title('Rastrigin Function')
% xlabel('x1')
% ylabel('x2')
% zlabel('fval')