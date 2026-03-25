%% PINN_DUAL_TARGET_FINAL.m
% -------------------------------------------------------------------------
% 1. Adatkinyerés 
% -------------------------------------------------------------------------
f_col = @(x) x(:); 
all_tout       = f_col(out_vx.Time);
all_sim_vx     = f_col(out_vx.Data);
all_sim_vy     = f_col(out_vy.Data);
all_sim_omega  = f_col(out_omega.Data);
all_sim_ay     = f_col(out_ay.Data);
all_sim_delta  = f_col(out_delta.Data);
all_sim_yawacc = f_col(out_yawacc.Data); 
all_sim_ax     = f_col(out_ax.Data);     

% -------------------------------------------------------------------------
% 2. Bemenetek és Targetek összeállítása
% -------------------------------------------------------------------------
% Bemenetek: [delta; vx; omega; ay; ax] 
inputs = [all_sim_delta, all_sim_vx, all_sim_omega, all_sim_ay, all_sim_ax]';

% Targetek: [vy; yawacc] 
targets = [all_sim_vy, all_sim_yawacc]';

% Normalizálás
mu_in = mean(inputs, 2); sigma_in = std(inputs, 0, 2) + 1e-6;
inputs_norm = (inputs - mu_in) ./ sigma_in;

dlInputs = dlarray(inputs_norm, 'CB');
dlTargets = dlarray(targets, 'CB');

% Pacejka paraméterek 
params.B = 25.046; params.C = 1.138; params.D = 8272.4;
params.m = 2108.0; params.Iz = 3954.288; params.lf = 1.47; params.lr = 1.50;

% -------------------------------------------------------------------------
% 3. Hálózat és Tanítás 
% -------------------------------------------------------------------------
net = dlnetwork([
    featureInputLayer(size(inputs, 1))
    fullyConnectedLayer(40) 
    tanhLayer
    fullyConnectedLayer(40)
    tanhLayer
    fullyConnectedLayer(2) 
]);

numEpochs = 2000; learningRate = 0.002; lambda = 0.05;
velocity = []; squaredGradient = [];

disp('PINN tanítása (v_y és YawAcc becslése)...');
tic;
for epoch = 1:numEpochs
    [loss, gradients] = dlfeval(@modelLoss, net, dlInputs, dlTargets, params, lambda, inputs);
    [net, velocity, squaredGradient] = adamupdate(net, gradients, velocity, squaredGradient, epoch, learningRate);
    if mod(epoch, 500) == 0 || epoch == 1
        fprintf('Epoch %4d | Loss: %.8f\n', epoch, extractdata(loss));
    end
end
toc;

% -------------------------------------------------------------------------
% 4. Plotolás 
% -------------------------------------------------------------------------
predictions = extractdata(forward(net, dlInputs));
vy_pred = predictions(1, :);
ya_pred = predictions(2, :);

% Vy plot
figure
plot(all_tout, targets(1,:), 'b', 'LineWidth', 2); hold on;
plot(all_tout, vy_pred, 'r--');
title('Vy (Oldalirányú sebesség) becslése'); ylabel('m/s'); grid on; legend('Ref','PINN');

% YawAcc plot
figure
plot(all_tout, targets(2,:), 'b', 'LineWidth', 2); hold on;
plot(all_tout, ya_pred, 'g--');
title('YawAcc (Legördülési gyorsulás) becslése'); ylabel('rad/s^2'); xlabel('Idő [s]'); grid on; legend('Ref','PINN');

% -------------------------------------------------------------------------
% 5. ModelLoss 
% -------------------------------------------------------------------------
function [loss, gradients] = modelLoss(net, dlInputs, dlTargets, p, lambda, raw_in)
    preds = forward(net, dlInputs);
    vy_pred = preds(1, :);
    ya_pred = preds(2, :);
    
    % Adat-alapú hiba 
    lossData = l2loss(vy_pred, dlTargets(1, :)) + l2loss(ya_pred, dlTargets(2, :));
    
    % Fizikai számítások
    delta = raw_in(1, :); vx = max(raw_in(2, :), 0.8); omega = raw_in(3, :); ay_meas = raw_in(4, :);
    
    af = delta - (vy_pred + (p.lf * omega)) ./ vx;
    ar = (p.lr * omega - vy_pred) ./ vx;
    
    Fyf = p.D * sin(p.C * atan(p.B * af));
    Fyr = p.D * sin(p.C * atan(p.B * ar));
    
    lossAccel = l2loss((Fyf + Fyr) / p.m, ay_meas);
    lossMomentum = l2loss((Fyf * p.lf - Fyr * p.lr) / p.Iz, ya_pred); 
    
    lossPhys = 0.5 * lossAccel + 0.5 * lossMomentum;
    loss = lossData + lambda * lossPhys;
    gradients = dlgradient(loss, net.Learnables);
end
% -------------------------------------------------------------------------
% 5. Trajektória Rekonstrukció 
% -------------------------------------------------------------------------
vy_pred_all = extractdata(forward(net, dlInputs)); 
% Ha a hálózatodnak 2 kimenete van, akkor a 2. sor a YawAcc 
ya_pred_all = vy_pred_all(2, :);
vy_pred_val = vy_pred_all(1, :);

Ts = mean(diff(all_tout));
numS = length(all_tout);

% Inicializálás 
X_p = zeros(numS, 1); Y_p = zeros(numS, 1); Psi_p = zeros(numS, 1);
X_ref = zeros(numS, 1); Y_ref = zeros(numS, 1); Psi_ref = zeros(numS, 1);

for t = 1:numS-1
    % PINN alapú integrálás 
    Psi_p(t+1) = Psi_p(t) + all_sim_omega(t) * Ts; 
    dX_p = (all_sim_vx(t)*cos(Psi_p(t)) - vy_pred_val(t)*sin(Psi_p(t))) * Ts;
    dY_p = (all_sim_vx(t)*sin(Psi_p(t)) + vy_pred_val(t)*cos(Psi_p(t))) * Ts;
    X_p(t+1) = X_p(t) + dX_p;
    Y_p(t+1) = Y_p(t) + dY_p;
    
    % Referencia (CarMaker) az összehasonlításhoz
    Psi_ref(t+1) = Psi_ref(t) + all_sim_omega(t) * Ts;
    dX_r = (all_sim_vx(t)*cos(Psi_ref(t)) - all_sim_vy(t)*sin(Psi_ref(t))) * Ts;
    dY_r = (all_sim_vx(t)*sin(Psi_ref(t)) + all_sim_vy(t)*cos(Psi_ref(t))) * Ts;
    X_ref(t+1) = X_ref(t) + dX_r;
    Y_ref(t+1) = Y_ref(t) + dY_r;
end

% -------------------------------------------------------------------------
% 6. Megjelenítés külön ablakokban 
% -------------------------------------------------------------------------

% 3. ÁBRA: Globális útvonal 
figure('Name', 'Globális Trajektória');
plot(X_ref, Y_ref, 'b', 'LineWidth', 2, 'DisplayName', 'Valódi út (Ref)'); hold on;
plot(X_p, Y_p, 'r--', 'LineWidth', 1.5, 'DisplayName', 'PINN rekonstruált út');
title('Jármű globális útvonala a pályán'); 
xlabel('Globális X [m]'); ylabel('Globális Y [m]'); 
axis equal; grid on; legend show;

save('my_pinn_model.mat', 'net', 'mu_in', 'sigma_in');