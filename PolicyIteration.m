% Policy Iteration

% Parameters
beta = 0.6;
eta_values = [0.9, 0.7, 0.01];
threshold = 1e-6;

% Transition probabilities
% let G=1, B=2, u=u+1 (since matlab cant have 0 for a matrix slot)
P = zeros(2, 2, 2);
P(1, 1, 2) = 0.1; P(2, 1, 2) = 0.9;
P(1, 1, 1) = 0.9; P(2, 1, 1) = 0.1;
P(1, 2, 2) = 0.5; P(2, 2, 2) = 0.5;
P(1, 2, 1) = 0.9; P(2, 2, 1) = 0.1;

% Iterate over eta values
for eta = eta_values
    % Initialize policy and value function
    gamma = ones(1, 2);
    V = zeros(1, 2);

    while true
        % Policy Evaluation
        while true
            V_prev = V;
            
            % Calculate current policy and value/cost
            for x = 1:2
                u = gamma(x);
                V(x) = cost(x, u, eta) + beta * sum(P(:, x, u) .* V_prev');
            end

            if max(abs(V - V_prev)) < threshold
                break;
            end
        end
        
        % Policy Improvement
        policy_stable = true;
        for x = 1:2
            % Store the old action from the current policy
            old_action = gamma(x);

            % Calculate the expected value for each action at state x and store in v_temp
            v_temp = [cost(x, 1, eta) + beta * sum(P(:, x, 1) .* V'), cost(x, 2, eta) + beta * sum(P(:, x, 2) .* V')];
    
            % Choose the action with the smallest expected value (minimizing cost)
            [~, new_action] = min(v_temp);

            % Update the policy with the new action
            gamma(x) = new_action;

            % Check if the policy has changed
            if old_action ~= new_action
                policy_stable = false;
            end
        end

        % Check for convergence
        if policy_stable
            break;
        end
    end

    % Print results
    disp(['Eta: ', num2str(eta)]);
    disp('Optimal Solution, V:');
    disp(V);
    disp('Optimal policy:');
    % Gamma equals 1 or 2 due to matlab notation so have to subtract 1
    disp(gamma - 1);
end

% Cost function as defined in the question
function cost = cost(x, u, eta)
    cost = -(x == 1 && u == 2) + eta * (u - 1);
end
