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