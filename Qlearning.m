% Q-Learning

% Parameters
eta = 0.7;
beta = 0.6;
threshold = 1e-6;
num_episodes = 5000;
max_steps = 100;

% Transition probabilities
% let G=1, B=2, u=u+1 (since matlab cant have 0 for a matrix slot)
P = zeros(2, 2, 2);
P(1, 1, 2) = 0.1; P(2, 1, 2) = 0.9;
P(1, 1, 1) = 0.9; P(2, 1, 1) = 0.1;
P(1, 2, 2) = 0.5; P(2, 2, 2) = 0.5;
P(1, 2, 1) = 0.9; P(2, 2, 1) = 0.1;

% Initialize Q function
Q = zeros(2, 2);
state_count = zeros(2, 2);

% Q-Learning
for episode = 1:num_episodes
    x = randi(2);
    
    for step = 1:max_steps
        % Choose action
        u = randi(2);
        
        % Update state count
        state_count(x, u) = state_count(x, u) + 1;
        
        % Learning rate
        alpha = 1 / (1 + state_count(x, u));
        
        % Calculate cost and next state
        reward = -(x == 1 && u == 2) + eta * (u - 1);
        x_next = find(mnrnd(1, squeeze(P(:, x, u))));
        
        % Update Q function
        Q(x, u) = Q(x, u) + alpha * (reward + beta * min(Q(x_next, :)) - Q(x, u));
        
        % Update state
        x = x_next;
    end
end

% Extract policy
gamma = zeros(1, 2);
[~, gamma] = min(Q, [], 1);

% Print results
disp('Q-Learning (Minimum)');
disp('Q values:');
disp(Q);
disp('Optimal policy:');
% Gamma equals 1 or 2 due to matlab notation so have to subtract 1
disp(gamma - 1);
