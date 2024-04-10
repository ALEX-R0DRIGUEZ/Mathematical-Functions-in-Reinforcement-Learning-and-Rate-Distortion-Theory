% Kalman Filter

% Initialize state and covariance
%x = zeros(3,1);           % Initial state vector
x = [-1; -1; -11];
Sigma = eye(3);           % Initial state covariance matrix
y = zeros(3,1);           % Initial measurement vector

% Define the system matrices
A = [2 1 0; 0 2 1; 0 0 2]; % State transition matrix
C = [4 0 0];               % Measurement matrix
Q = eye(3);                % Process noise covariance matrix
R = 1;                     % Measurement noise covariance matrix
T = 1000;                  % Number of time steps

% Arrays for storing results
xt = zeros(3, T);         % True state history
yt = zeros(3, T);         % Measurement history
mt = zeros(3, T);         % State estimate history
mt(:,1) = zeros(3,1);     % Initial state estimate
et = zeros(3, T);         % State estimation error history

% Generate true state and measurements
for t = 1:T
    % Store true state and observed measurement vectors
    xt(:,t) = x;
    yt(:,t) = y;
    
    % Generate process and measurement noise
    w = [normrnd(0,1); normrnd(0,1); normrnd(0,1)];
    v = normrnd(0,1);
    
    % Update true state and observed measurement vectors
    x = A * x + w;
    y = C * x + v;
end

% Estimate state using Kalman filter
for t = 2:T
    % Update state estimate using Kalman filter equations
    K_gain = A * Sigma * C' / (C * Sigma * C' + R);
    mt(:,t) = A * mt(:,t-1) + K_gain * (yt(t-1) - C * mt(:,t-1));
    Sigma = A * Sigma * A' - K_gain * C * Sigma * A' + Q;
    
    % Calculate state estimation error
    et(:,t) = xt(:,t) - mt(:,t);
end

% Check Controllability and Observability
B_ctrl = eye(3);
Cm_ctrl = ctrb(A, B_ctrl);
rank(Cm_ctrl)

Om_obs = obsv(A, C);
rank(Om_obs)

% Plot True State, xt
figure;
plot(xt');
title('True state, xt');
xlabel('Time');
ylabel('State');
legend('x1', 'x2', 'x3');

% Plot Estimated State, mt
figure;
plot(mt');
title('Estimated state, mt');
xlabel('Time');
ylabel('State');
legend('x1', 'x2', 'x3');

% Plot State Estimation Error, xt-mt
figure;
plot(et');
title('State estimation error, xt-mt');
xlabel('Time');
ylabel('Error');
legend('x1', 'x2', 'x3');

