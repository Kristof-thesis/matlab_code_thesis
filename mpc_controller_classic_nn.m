function u_out = mpc_controller_classic_nn(input_vector)
    % --- HAGYOMÁNYOS NEURÁLIS HÁLÓ ALAPÚ MPC ---
    % Mux sorrend: [vy, omega, y_global, psi, u_prev, t_curr, vx_meas]
    
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
    vx_meas = max(input_vector(7), 1.0); 
    
    % --- 2. Konfiguráció és Inicializálás ---
    N = 60;         
    Ts = 0.02;      
    
    if isempty(solver_impl)
        casadi_path = 'C:\CM_Projects\Test_1\src_cm4sl\casadi-3.7.2-windows64-matlab2018b';
        if ~contains(path, casadi_path)
            addpath(casadi_path);
        end
        
        import casadi.*
        
        % --- ITT A HAGYOMÁNYOS NN ADATOKAT OLVASSUK BE ---
        nn_net = evalin('base', 'nn_net');
        mu_in = evalin('base', 'mu_in_nn');        
        sigma_in = evalin('base', 'sigma_in_nn');  
        
        % =================================================================
        % JAVÍTOTT RÉSZ: Manuális és fix súly-kicsomagolás az indexhiba ellen
        % =================================================================
        W = {}; b = {};
        
        % 1. Réteg (Bemenet -> 1. Rejtett réteg: 15 neuron)
        W{1} = casadi.MX(double(nn_net.IW{1,1})); 
        b{1} = casadi.MX(double(nn_net.b{1}));
        
        % 2. Réteg (1. Rejtett -> 2. Rejtett réteg: 10 neuron)
        W{2} = casadi.MX(double(nn_net.LW{2,1})); 
        b{2} = casadi.MX(double(nn_net.b{2}));
        
        % 3. Réteg (2. Rejtett -> Kimeneti réteg: 2 kimenet)
        W{3} = casadi.MX(double(nn_net.LW{3,2})); 
        b{3} = casadi.MX(double(nn_net.b{3}));
        % =================================================================
        
        % Szimbolikus NN funkció felépítése
        u_s = casadi.MX.sym('u'); 
        vx_s = casadi.MX.sym('vx'); 
        x_v_s = casadi.MX.sym('x', 4);
        
        % Bemeneti vektor (Kormányjel, sebesség, szögsebesség, vx*omega, konstans 0)
        raw_in = vertcat(u_s, vx_s, x_v_s(2), vx_s * x_v_s(2), 0); 
        h = (raw_in - casadi.MX(mu_in)) ./ casadi.MX(sigma_in);
        
        % Előrecsatolás a 3 rétegen keresztül
        h = tanh(W{1} * h + b{1}); 
        h = tanh(W{2} * h + b{2});
        pinn_out = W{3} * h + b{3}; % Utolsó rétegre nem teszünk tanh-t
        
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
        
        % --- Ugyanaz a szinuszos trajektória a fair összehasonlításhoz ---
        amplitude = 3.0;  
        frequency = 0.2;  
        t_start = 1.0;    
        
        for k = 1:N
            curr_u = curr_u + dU(k);
            t_f = t_s + k * Ts;
            
            ref_y = if_else(t_f < t_start, 0.0, ...
                            amplitude * sin(2 * pi * frequency * (t_f - t_start)));
            
            ref_psi = if_else(t_f < t_start, 0.0, ...
                              (amplitude * 2 * pi * frequency * cos(2 * pi * frequency * (t_f - t_start))) / vx_p_s);
            
            res = nn_func(curr_u, vx_p_s, curr_x);
            vy_next = res(1);
            yaw_acc = res(2); 
            
            % Diszkrét dinamika predikció
            omega_next = curr_x(2) + yaw_acc * Ts;
            y_next     = curr_x(3) + (vx_p_s * sin(curr_x(4)) + vy_next * cos(curr_x(4))) * Ts;
            psi_next   = curr_x(4) + curr_x(2) * Ts;
            
            curr_x = vertcat(vy_next, omega_next, y_next, psi_next);
            
            % TÖKÉLETESEN UGYANAZOK A SÚLYOK!
            obj = obj + 1500 * (curr_x(3) - ref_y)^2;     
            obj = obj + 800 * (curr_x(4) - ref_psi)^2;   
            obj = obj + 2000 * dU(k)^2;                   
        end
        obj = obj + 3000 * (curr_x(3) - ref_y)^2;
        
        nlp = struct('x', dU, 'f', obj, 'p', vertcat(x0_s, u_p_s, t_s, vx_p_s));
        opts = struct('ipopt', struct('max_iter', 30, 'print_level', 0, 'tol', 1e-3), 'print_time', 0);
        solver_impl = casadi.nlpsol('solver', 'ipopt', nlp, opts);
        last_du_guess = zeros(N, 1);
    end
    
    p_values = [x_meas; u_prev; t_curr; vx_meas];
    sol = solver_impl('x0', last_du_guess, 'p', p_values, 'lbx', -0.015, 'ubx', 0.015);
    du_opt = full(sol.x);
    last_du_guess = [du_opt(2:end); du_opt(end)]; 
    
    u_out = max(min(u_prev + du_opt(1), 0.4), -0.4);
end