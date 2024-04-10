% Value Iteration

% Parameters
beta = 0.6;
eta_values = [0.9, 0.7, 0.01];
threshold = 1e-6;

% Transition probabilities
% let G=1, B=2, u=u+1 (since matlab cant have 0 for a matrix slot)
P = zeros(2, 2, 2); % P(xt+1, xt, ut)
P(1, 1, 2) = 0.1; P(2, 1, 2) = 0.9;
P(1, 1, 1) = 0.9; P(2, 1, 1) = 0.1;
P(1, 2, 2) = 0.5; P(2, 2, 2) = 0.5;
P(1, 2, 1) = 0.9; P(2, 2, 1) = 0.1;

% Iterate over eta values
for eta = eta_values
    % Initialize value function
    V = zeros(1, 2);
    
    while true
        % Update value function
        vPrev = V;
        
        % Calculate value
        v0 = [cost(2, 1, eta) + beta * sum(vPrev .* P(2, :, 1)), cost(2, 2, eta) + beta * sum(vPrev .* P(2, :, 2))];
        v1 = [cost(1, 1, eta) + beta * sum(vPrev .* P(1, :, 1)), cost(1, 2, eta) + beta * sum(vPrev .* P(1, :, 2))];
        
        V = [min(v0),min(v1)];
        
        % Check for convergence
        if max(abs(V - vPrev)) < threshold
            break;
        end
    end
    
    % Compute optimal policy
    gamma = zeros(1, 2);
    [~,gamma(1)] = min(v0);
    [~,gamma(2)] = min(v1);
    % Print results
    disp(['Eta: ', num2str(eta)]);
    disp('Optimal Solution, V:');
    disp(V);
    disp('Optimal policy:');
    % Gamma equals 1 or 2 due to matlab notation so have to subtract 1
    disp(gamma-1);
end

% Cost function as defined in the question
function cost = cost(x,u,eta)
    cost = -(x == 1 && u == 2) + eta * (u - 1);
end