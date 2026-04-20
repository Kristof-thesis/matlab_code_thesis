%% TELJES CASADI-PINN-MPC RENDSZER 
% Verzió: MX-alapú, Warm-start,
CASADI_PATH = 'C:\CM_Projects\Test_1\src_cm4sl\casadi-3.7.2-windows64-matlab2018b';

% 1. Hozzáadás a path-hoz
if ~contains(path, CASADI_PATH)
    addpath(CASADI_PATH);
end

% 2. AZONNALI ellenőrzés
if exist('casadi.MX', 'class')
    import casadi.*
    disp('CasADi sikeresen betöltve és importálva!');
else
    error('Hiba: A megadott útvonalon nem található a CasADi!');
end

test_var = MX.sym('test');

%% 2. SZIMBOLIKUS PINN HÁLÓ ÉPÍTÉSE 
fprintf('PINN háló konvertálása szimbolikus formába (MX)...\n');

layers = pinn_net.Layers;
fc_idx = find(arrayfun(@(l) isa(l, 'nnet.cnn.layer.FullyConnectedLayer'), layers));

W = {}; b = {};
for i = 1:length(fc_idx)
    W{i} = casadi.MX(double(layers(fc_idx(i)).Weights)); 
    b{i} = casadi.MX(double(layers(fc_idx(i)).Bias));
end

% Szimbolikus bemenetek (MX)
u_s  = casadi.MX.sym('u');      
vx_s = casadi.MX.sym('vx');    
x_s  = casadi.MX.sym('x', 4);   

% Normalizáló konstansok MX típussá alakítása
mu_mx = casadi.MX(mu_in);
sig_mx = casadi.MX(sigma_in);

% Normalizálás és Forward Pass
raw_in = vertcat(u_s, vx_s, x_s(2), vx_s * x_s(2), 0); 
norm_in = (raw_in - mu_mx) ./ sig_mx;

h = norm_in;
for i = 1:length(W)-1
    h = tanh(W{i} * h + b{i}); 
end
pinn_out = W{end} * h + b{end}; 

% CasADi függvény létrehozása
nn_func = casadi.Function('nn_func', {u_s, vx_s, x_s}, {pinn_out});

%% 3. MPC OPTIMALIZÁLÓ (IPOPT) ÖSSZEÁLLÍTÁSA
fprintf('MPC solver konfigurálása...\n');

N = 60;         
Ts = 0.01;      
vx_const = 15;  

% Döntési változók és paraméterek
dU = casadi.MX.sym('dU', N);   
u_prev_s = casadi.MX.sym('u_p');
x0_s = casadi.MX.sym('x0', 4);
t_curr_s = casadi.MX.sym('t_curr'); 

obj = 0;
curr_x = x0_s;
curr_u = u_prev_s;

for k = 1:N
    curr_u = curr_u + dU(k);
    t_future = t_curr_s + k * Ts; 
    
    
    ref_step = if_else(t_future < 1.0, 0.0, 3.0); 
    
    res = nn_func(curr_u, vx_const, curr_x);
    vy_next = res(1);
    yaw_acc = res(2);
    
    omega_next = curr_x(2) + yaw_acc * Ts;
    y_next     = curr_x(3) + (vx_const * sin(curr_x(4)) + vy_next * cos(curr_x(4))) * Ts;
    psi_next   = curr_x(4) + curr_x(2) * Ts;
    
    curr_x = vertcat(vy_next, omega_next, y_next, psi_next);
    
    obj = obj + 300 * (curr_x(3) - ref_step)^2;   
    obj = obj + 1000 * (curr_x(4))^2;             
    obj = obj + 100 * dU(k)^2;                    
end

obj = obj + 5000 * (curr_x(3) - ref_step)^2;
nlp = struct('x', dU, 'f', obj, 'p', vertcat(x0_s, u_prev_s, t_curr_s));

opts = struct;
opts.ipopt.max_iter = 25;
opts.ipopt.print_level = 0;
opts.print_time = 0;
opts.ipopt.hessian_approximation = 'limited-memory'; 
opts.ipopt.warm_start_init_point = 'yes';           
opts.ipopt.tol = 1e-3;

solver = casadi.nlpsol('solver', 'ipopt', nlp, opts);

%% 4. ZÁRT HURKÚ SZIMULÁCIÓ
fprintf('Szimuláció indítása...\n');
Tsim = 600;
x_curr = [0; 0; 0; 0]; 
u_last = 0;
du_guess = zeros(N, 1); 

X_history = zeros(4, Tsim);
U_history = zeros(1, Tsim);
REF_history = zeros(1, Tsim);

tic;
for t = 1:Tsim
    t_now = (t-1) * Ts;
    
    % Paraméterek: [állapot; előző_kormány; aktuális_idő]
    p_values = [x_curr; u_last; t_now];
    
    sol = solver('x0',  du_guess, ...
                 'p',   p_values, ...
                 'lbx', -0.015 * ones(N,1), ...
                 'ubx',  0.015 * ones(N,1));
    
    du_opt_full = full(sol.x);
    du_guess = du_opt_full; 
    
    u_last = u_last + du_opt_full(1);
    u_last = max(min(u_last, 0.4), -0.4); 
    
    % Valós rendszer szimulálása
    res_sim = nn_func(u_last, vx_const, x_curr);
    vy_n = full(res_sim(1));
    om_n = x_curr(2) + full(res_sim(2)) * Ts;
    y_n  = x_curr(3) + (vx_const * sin(x_curr(4)) + vy_n * cos(x_curr(4))) * Ts;
    ps_n = x_curr(4) + x_curr(2) * Ts;
    
    x_curr = [vy_n; om_n; y_n; ps_n];
    
    X_history(:,t) = x_curr;
    U_history(t) = u_last;
      
    if t_now < 1.0
        REF_history(t) = 0;
    else
        REF_history(t) = 3.0;
    end
    
    if mod(t,50)==0, fprintf('Lépés: %d/%d\n', t, Tsim); end
end
sim_time = toc;
fprintf('Szimuláció kész! Időtartam: %.4f mp\n', sim_time);

%% 5. MEGJELENÍTÉS (Frissítve a dinamikus referenciával)
time = (0:Tsim-1) * Ts;
figure('Color', 'w', 'Name', 'MPC Pályakövetés');
subplot(2,1,1);
plot(time, X_history(3,:), 'b', 'LineWidth', 2); hold on;
plot(time, REF_history, '--r', 'LineWidth', 1.5); % Dinamikus referencia vonal
grid on; ylabel('Y pozíció [m]'); title('CasADi-PINN-MPC: Sávváltás teszt');
legend('Jármű útja', 'Referencia');

subplot(2,1,2);
plot(time, rad2deg(U_history), 'm', 'LineWidth', 2);
grid on; ylabel('Kormányszög [deg]'); xlabel('Idő [s]');