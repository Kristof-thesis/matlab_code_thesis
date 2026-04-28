function steer_cmd = Analytic_MPC_Controller(vx, vy, omega, tx_glob, ty_glob, psi)
% #########################################################################
% ANALITIKUS MPC - Bicikli modell + Pacejka 
% #########################################################################

persistent ref_traj is_initialized last_steer s_total y0 psi0
Ts = 0.02;
N = 40;
steer_ratio = 15;

% JÁRMŰ PARAMÉTEREK 
m = 2108.0; Iz = 3954.0; lf = 1.47; lr = 1.50;

B = 25.0; C = 1.13; D = 8272.0;

if isempty(is_initialized)
    R = load('lane_change_reference.mat');
    ref_traj = R.ref_matrix;
    y0 = ty_glob; 
    psi0 = psi;
    last_steer = 0.0;
    s_total = 0.0;
    is_initialized = true;
end

% Relatív pozíciók
y_rel = ty_glob - y0;
psi_rel = psi - psi0;
s_total = s_total + vx * Ts;

best_steer_delta = 0.0;
min_cost = 1e10; 
candidate_deltas = [-0.01, -0.005, 0, 0.005, 0.01];

for i = 1:5
    curr_delta = candidate_deltas(i);
    current_steer_rad = (last_steer + curr_delta); % Kerékszög [rad]
    
    % Predikciós állapotok
    t_y = y_rel; t_psi = psi_rel; t_vy = vy; t_om = omega; t_s = s_total;
    cost = 0.0;
    
    for k = 1:N
        % 1. Csúszásszögek
        v_safe = max(vx, 0.5);
        af = current_steer_rad - (t_vy + lf * t_om) / v_safe;
        ar = (lr * t_om - t_vy) / v_safe;
        
        % 2. Oldalerők 
        Fyf = D * sin(C * atan(B * af));
        Fyr = D * sin(C * atan(B * ar));
        
        % 3. Dinamika 
        dvy = (Fyf + Fyr) / m - vx * t_om;
        dom = (lf * Fyf - lr * Fyr) / Iz;
        
        % 4. Integrálás 
        t_vy = t_vy + dvy * Ts;
        t_om = t_om + dom * Ts;
        t_psi = t_psi + t_om * Ts;
        t_y = t_y + (vx * sin(t_psi) + t_vy * cos(t_psi)) * Ts;
        t_s = t_s + vx * Ts;
        
        % 5. Költség 
        [~, idx] = min(abs(ref_traj(:,1) - t_s));
        cost = cost + 600 * (t_y - ref_traj(idx, 2))^2 + 300 * t_psi^2;
    end
    
    if cost < min_cost
        min_cost = cost;
        best_steer_delta = curr_delta;
    end
end

last_steer = last_steer + best_steer_delta;
last_steer = max(min(last_steer, 0.45), -0.45);

% Kimenet áttétellel
steer_cmd = last_steer * steer_ratio; 
end