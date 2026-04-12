%% ================= TELJES NN-MPC RENDSZER (0.01s / 100Hz) =================
% 1. Adatok betöltése
load('trained_pinn_data.mat'); 

% 2. Paraméterek hangolása
params.N  = 40;        
params.Ts = 0.01;      
params.vx = 10;        

% Q mátrix: [vy, omega, y, psi]
params.Q  = diag([1, 1, 150, 80]); 
params.Qf = diag([1, 1, 300, 150]); 
params.R  = 5;         

params.du_max = 0.01;  
params.u_max  = 0.4;

%% ================= SZIMULÁCIÓS CIKLUS =================
x = [0; 0; 0; 0];      
u = 0;                 
ref = [0; 0; 2; 0];    
Tsim = 100;            

X_log = zeros(4, Tsim);
U_log = zeros(1, Tsim);

fprintf('NN-MPC indítása (Max 5 iteráció / lépés)...\n');
tic;
for t = 1:Tsim
    
    u = mpc_controller(x, u, pinn_net, ref, params, mu_in, sigma_in);
    
    
    x = update_system(pinn_net, x, u, params, mu_in, sigma_in);
    
    X_log(:,t) = x;
    U_log(t)   = u;
    
    if mod(t,20) == 0, fprintf('Lépés: %d/%d\n', t, Tsim); end
end
toc;

%% ================= MEGJELENÍTÉS =================
time = (0:Tsim-1) * params.Ts;
figure('Color', 'w', 'Name', 'Optimalizált NN-MPC');

subplot(3,1,1);
plot(time, X_log(3,:), 'b', 'LineWidth', 2); hold on;
yline(ref(3), '--r', 'Cél');
ylabel('Pozíció (y) [m]'); grid on;

subplot(3,1,2);
plot(time, X_log(4,:)*180/pi, 'g', 'LineWidth', 2);
ylabel('Irányszög (\psi) [deg]'); grid on;

subplot(3,1,3);
plot(time, U_log*180/pi, 'm', 'LineWidth', 2);
ylabel('Kormány (\delta) [deg]'); xlabel('Idő [s]'); grid on;

%% ================= MPC VEZÉRLŐ FÜGGVÉNY =================
function u_opt = mpc_controller(x0, u_prev, net, ref, params, mu_in, sigma_in)
    N = params.N;
    dU0 = zeros(N,1);
    lb = -params.du_max * ones(N,1);
    ub =  params.du_max * ones(N,1);
    
    % GYORSÍTÁS
    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'Algorithm', 'sqp', ...
        'MaxIterations', 5, ... 
        'MaxFunctionEvaluations', 200);
    
    cost_fun = @(dU) cost_function(dU, x0, u_prev, net, ref, params, mu_in, sigma_in);
    dU_opt = fmincon(cost_fun, dU0, [], [], [], [], lb, ub, [], options);
    
    u_opt = u_prev + dU_opt(1);
    u_opt = max(min(u_opt, params.u_max), -params.u_max);
end

%% ================= KÖLTSÉGFÜGGVÉNY =================
function J = cost_function(dU, x0, u_prev, net, ref, params, mu_in, sigma_in)
    x = x0;
    u = u_prev;
    J = 0;
    for k = 1:params.N
        u = u + dU(k);
        u = max(min(u, params.u_max), -params.u_max);
        
        
        x = update_system(net, x, u, params, mu_in, sigma_in);
        
        error = x - ref;
        
        J = J + error' * params.Q * error + dU(k)^2 * params.R;
    end
    
    J = J + (x - ref)' * params.Qf * (x - ref);
end

%% ================= MODEL / UPDATE FUNCTION =================
function x_next = update_system(net, x, u, params, mu_in, sigma_in)
    vx = params.vx;
    omega = x(2);
    psi   = x(4);
    
    
    ay_approx = vx * omega;
    raw_in = [u; vx; omega; ay_approx; 0];
    norm_in = (raw_in - mu_in) ./ sigma_in;
    
   
    dlY = forward(net, dlarray(single(norm_in), 'CB'));
    y_nn = extractdata(dlY);
    
    vy_new = y_nn(1);
    yaw_acc = y_nn(2);
    
   
    x_next = zeros(4,1);
    x_next(1) = vy_new; 
    x_next(2) = x(2) + params.Ts * yaw_acc;
    x_next(3) = x(3) + params.Ts * (vx * sin(psi) + vy_new * cos(psi));
    x_next(4) = x(4) + params.Ts * x(2);
end