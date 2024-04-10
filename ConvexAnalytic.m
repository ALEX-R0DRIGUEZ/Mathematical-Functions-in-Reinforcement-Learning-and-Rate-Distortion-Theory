% Convex Analytic Method

% Parameters
eta = 0.7;

% Transition probabilities
% let G=1, B=2, u=u+1 (since matlab cant have 0 for a matrix slot)
P = zeros(2, 2, 2);
P(1, 1, 2) = 0.1; P(2, 1, 2) = 0.9;
P(1, 1, 1) = 0.9; P(2, 1, 1) = 0.1;
P(1, 2, 2) = 0.5; P(2, 2, 2) = 0.5;
P(1, 2, 1) = 0.9; P(2, 2, 1) = 0.1;

% Cost vector
c = [cost(2, 1, eta), cost(2, 2, eta), cost(1, 1, eta), cost(1, 2, eta)];

% Equality constraint matrices
Z = [1, 1, 0, 0; 0, 0, 1, 1];
X = [P(2, 2, 1),P(2, 2, 2),P(2, 1, 1),P(2, 1, 2);
     P(1, 2, 1),P(1, 2, 2),P(1, 1, 1),P(1, 1, 2)];
A = vertcat((Z - X), ones(1, 4));

% Equality constraint vector
b = [0, 0, 1];

% Lower bounds
lb = zeros(1, 4);

% Solve linear program
options = optimset('Algorithm', 'dual-simplex', 'Display', 'off');
u_opt = linprog(c, [], [], A, b, lb, [], options);

% Calculate Ecost
Ecost = c * u_opt;

% Extract optimal policy
gamma = [u_opt(1,1)/(u_opt(1,1)+u_opt(2,1)), u_opt(3,1)/(u_opt(3,1)+u_opt(4,1))];

% Print results
disp('Linear programming');
disp('Optimal Solution');
disp(Ecost);
disp('Optimal policy:');
disp(gamma);

% Cost function as defined in the question
function cost = cost(x, u, eta)
    cost = -(x == 1 && u == 2) + eta * (u - 1);
end
