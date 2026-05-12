function u_out = mpc_controller(input_vector)
    % --- Bemenetek felbontása (Mux sorrendje) ---
    % 1-2: vy, omega
    % 3:   y_global (CarMaker nyers Y koordináta)
    % 4:   psi (heading)
    % 5:   u_prev (visszacsatolt kormányjel)
    % 6:   t_curr (idő)
    % 7:   vx_meas (CarMaker sebesség)

    % Perzisztens változók az inicializáláshoz és a nullázáshoz
    persistent solver_impl last_du_guess y_start
    
    % --- 1. Relatív pozíció számítása ---
    y_global = input_vector(3);
    if isempty(y_start)
        y_start = y_global; 
    end
    y_rel = y_global - y_start; 
    
    x_meas = [input_vector(1); input_vector(2); y_rel; input_vector(4)];
    u_prev = input_vector(5);
    t_curr = input_vector(6);
    vx_meas = input_vector(7);

    % --- 2. Konfiguráció és Inicializálás ---
    N = 60;         
    Ts = 0.02;      
    
    if isempty(solver_impl)
        % --- ELÉRÉSI ÚT KEZELÉSE ---
        % Ellenőrizd, hogy ez az útvonal pontos-e a gépeden!
        casadi_path = 'C:\CM_Projects\Test_1\src_cm4sl\casadi-3.7.2-windows64-matlab2018b';
        if ~contains(path, casadi_path)
            addpath(casadi_path);
        end
        
        import casadi.*

        % Adatok beolvasása a Workspace-ből
        pinn_net = evalin('base', 'pinn_net');
        mu_in = evalin('base', 'mu_in');
        sigma_in = evalin('base', 'sigma_in');
        
        layers = pinn_net.Layers;
        fc_idx = find(arrayfun(@(l) isa(l, 'nnet.cnn.layer.FullyConnectedLayer'), layers));
        W = {}; b = {};
        for i = 1:length(fc_idx)
            W{i} = casadi.MX(double(layers(fc_idx(i)).Weights)); 
            b{i} = casadi.MX(double(layers(fc_idx(i)).Bias));
        end

        % Szimbolikus PINN funkció (Explicit casadi.MX hívásokkal)
        u_s = casadi.MX.sym('u'); 
        vx_s = casadi.MX.sym('vx'); 
        x_v_s = casadi.MX.sym('x', 4);
        
        raw_in = vertcat(u_s, vx_s, x_v_s(2), vx_s * x_v_s(2), 0); 
        h = (raw_in - casadi.MX(mu_in)) ./ casadi.MX(sigma_in);
        for i = 1:length(W)-1
            h = tanh(W{i} * h + b{i}); 
        end
        pinn_out = W{end} * h + b{end};
        nn_func = casadi.Function('nn_func', {u_s, vx_s, x_v_s}, {pinn_out});

        % --- MPC Struktúra ---
        dU = casadi.MX.sym('dU', N);
        u_p_s = casadi.MX.sym('u_p');
        x0_s = casadi.MX.sym('x0', 4);
        t_s = casadi.MX.sym('t_curr');
        vx_p_s = casadi.MX.sym('vx_p');
        
        obj = 0;
        curr_x = x0_s;
        curr_u = u_p_s;
        
        % Sávváltás paraméterei
        t_start = 1.0; t_maneuver = 3.0; lane_width = 3.5;

        for k = 1:N
            curr_u = curr_u + dU(k);
            t_f = t_s + k * Ts;
            
            % Szinuszos referencia trajektória
            ref_y = if_else(t_f < t_start, 0.0, ...
                    if_else(t_f > (t_start + t_maneuver), lane_width, ...
                    lane_width * (1 - cos(pi * (t_f - t_start) / t_maneuver)) / 2));
            
            res = nn_func(curr_u, vx_p_s, curr_x);
            vy_next = res(1);
            yaw_acc = res(2);
            
            % Diszkrét dinamika
            omega_next = curr_x(2) + yaw_acc * Ts;
            y_next     = curr_x(3) + (vx_p_s * sin(curr_x(4)) + vy_next * cos(curr_x(4))) * Ts;
            psi_next   = curr_x(4) + curr_x(2) * Ts;
            
            curr_x = vertcat(vy_next, omega_next, y_next, psi_next);
            
            % Költségfüggvény
            obj = obj + 300 * (curr_x(3) - ref_y)^2;   
            obj = obj + 1000 * (curr_x(4))^2;             
            obj = obj + 100 * dU(k)^2;                    
        end
        obj = obj + 5000 * (curr_x(3) - ref_y)^2;

        nlp = struct('x', dU, 'f', obj, 'p', vertcat(x0_s, u_p_s, t_s, vx_p_s));
        opts = struct('ipopt', struct('max_iter', 25, 'print_level', 0, 'tol', 1e-3), 'print_time', 0);
        solver_impl = casadi.nlpsol('solver', 'ipopt', nlp, opts);
        last_du_guess = zeros(N, 1);
    end

    % --- 3. Optimalizálás az aktuális lépésben ---
    p_values = [x_meas; u_prev; t_curr; vx_meas];
    
    sol = solver_impl('x0', last_du_guess, 'p', p_values, ...
                      'lbx', -0.015, 'ubx', 0.015);
    du_opt = full(sol.x);
    last_du_guess = du_opt; 
    
    u_out = max(min(u_prev + du_opt(1), 0.4), -0.4);
end