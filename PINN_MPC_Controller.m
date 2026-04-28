function steer_cmd = PINN_MPC_Controller(vx, vy, omega, tx_glob, ty_glob, psi)
% #########################################################################
% PINN-MPC Járműirányító 
% #########################################################################

persistent net_weights ref_traj is_initialized last_steer s_total y0 psi0
Ts = 0.02;
N = 40;
steer_ratio = 15;

if isempty(is_initialized)
    % Adatok betöltése
    S = load('trained_pinn_params.mat');
    net_weights = S.pinn_params;
    
    R = load('lane_change_reference.mat');
    ref_traj = R.ref_matrix;
    
    %Az első időlépés értékeit vesszük alapul
    y0 = ty_glob;      
    psi0 = psi;   % Kezdeti irány
    
    last_steer = 0.0;
    s_total = 0.0;
    is_initialized = true;
end

% RELATÍV ÉRTÉKEK SZÁMÍTÁSA
% Így a PINN és a referencia táblázat is 0-ról induló értékeket kap
y_rel = ty_glob - y0;
psi_rel = psi - psi0;

% Hosszirányú út becslése 
s_total = s_total + vx * Ts;

best_steer_delta = 0.0;
min_cost = 1e10; 
candidate_deltas = [-0.01, -0.005, 0, 0.005, 0.01];

for i = 1:5
    curr_delta = candidate_deltas(i);
    current_steer = last_steer + curr_delta;
    
    % Predikciós állapotok 
    temp_y = y_rel;
    temp_psi = psi_rel;
    temp_vy = vy;
    temp_omega = omega;
    temp_s = s_total;
    cost = 0.0;
    
    for k = 1:N
        % PINN Modell
        in_raw = [current_steer; vx; temp_omega; 0.0; 0.0];
        in_norm = (in_raw - net_weights.mu) ./ net_weights.sigma;
        
        h1 = tanh(net_weights.W1 * in_norm + net_weights.b1);
        h2 = tanh(net_weights.W2 * h1 + net_weights.b2);
        out = net_weights.W3 * h2 + net_weights.b3;
        
        vy_pred = out(1);
        yaw_acc = out(2);
        
        % Integrálás 
        temp_omega = temp_omega + yaw_acc * Ts;
        temp_psi   = temp_psi + temp_omega * Ts;
        temp_y     = temp_y + (vx * sin(temp_psi) + vy_pred * cos(temp_psi)) * Ts;
        temp_s     = temp_s + vx * Ts;
        
        % Referencia keresés 
        [~, idx] = min(abs(ref_traj(:,1) - temp_s));
        y_ref_k = ref_traj(idx, 2);
        
        % Hiba számítás
        cost = cost + 600 * (temp_y - y_ref_k)^2 + 300 * temp_psi^2;
    end
    
    if cost < min_cost
        min_cost = cost;
        best_steer_delta = curr_delta;
    end
end

u_steer = last_steer + best_steer_delta;
u_steer = max(min(u_steer, 0.45), -0.45);
last_steer = u_steer;

steer_cmd = u_steer * steer_ratio; 
end