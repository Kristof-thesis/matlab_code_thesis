%% PINN_TRAIN_AND_EXPORT_FINAL.m
% -------------------------------------------------------------------------
% 1. Adatkinyerés (CarMaker kimenetek alapján)
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
inputs = [all_sim_delta, all_sim_vx, all_sim_omega, all_sim_ay, all_sim_ax]';
targets = [all_sim_vy, all_sim_yawacc]';

% Normalizálás
mu_in = mean(inputs, 2); 
sigma_in = std(inputs, 0, 2) + 1e-6;
inputs_norm = (inputs - mu_in) ./ sigma_in;

dlInputs = dlarray(inputs_norm, 'CB');
dlTargets = dlarray(targets, 'CB');

% Pacejka és Jármű paraméterek
params.B = 25.046; params.C = 1.138; params.D = 8272.4; 
params.m = 2108.0; params.Iz = 3954.288; params.lf = 1.47; params.lr = 1.50;

% -------------------------------------------------------------------------
% 3. Hálózat felépítése és Tanítás 
% -------------------------------------------------------------------------
net = dlnetwork([
    featureInputLayer(size(inputs, 1), 'Name', 'input')
    fullyConnectedLayer(40, 'Name', 'fc1') 
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(40, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(2, 'Name', 'output') 
]);

numEpochs = 2000; learningRate = 0.002; lambda = 0.05;
velocity = []; squaredGradient = [];

disp('PINN tanítása folyamatban...');
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
% 4. Predikciók kinyerése
% -------------------------------------------------------------------------
predictions = extractdata(forward(net, dlInputs));
vy_pred_val = predictions(1, :);
ya_pred_val = predictions(2, :);

% -------------------------------------------------------------------------
% 5. Trajektória Rekonstrukció 
% -------------------------------------------------------------------------
Ts = mean(diff(all_tout));
numS = length(all_tout);
X_p = zeros(numS, 1); Y_p = zeros(numS, 1); Psi_p = zeros(numS, 1);
X_ref = zeros(numS, 1); Y_ref = zeros(numS, 1); Psi_ref = zeros(numS, 1);

for t = 1:numS-1
    % PINN alapú integrálás 
    Psi_p(t+1) = Psi_p(t) + all_sim_omega(t) * Ts; 
    X_p(t+1) = X_p(t) + (all_sim_vx(t)*cos(Psi_p(t)) - vy_pred_val(t)*sin(Psi_p(t))) * Ts;
    Y_p(t+1) = Y_p(t) + (all_sim_vx(t)*sin(Psi_p(t)) + vy_pred_val(t)*cos(Psi_p(t))) * Ts;
    
    % Referencia (CarMaker)
    Psi_ref(t+1) = Psi_ref(t) + all_sim_omega(t) * Ts;
    X_ref(t+1) = X_ref(t) + (all_sim_vx(t)*cos(Psi_ref(t)) - all_sim_vy(t)*sin(Psi_ref(t))) * Ts;
    Y_ref(t+1) = Y_ref(t) + (all_sim_vx(t)*sin(Psi_ref(t)) + all_sim_vy(t)*cos(Psi_ref(t))) * Ts;
end

% -------------------------------------------------------------------------
% 6. MEGJELENÍTÉS KÜLÖN FIGURE-ÖKBEN
% -------------------------------------------------------------------------

% 1. ÁBRA: Vy összehasonlítás
figure('Name', 'Oldalirányú sebesség (Vy)');
plot(all_tout, targets(1,:), 'b', 'LineWidth', 2); hold on;
plot(all_tout, vy_pred_val, 'r--');
title('Vy (Oldalirányú sebesség) becslése'); ylabel('m/s'); xlabel('Idő [s]');
grid on; legend('CarMaker (Ref)', 'PINN Predikció');

% 2. ÁBRA: YawAcc összehasonlítás
figure('Name', 'Legördülési gyorsulás (YawAcc)');
plot(all_tout, targets(2,:), 'b', 'LineWidth', 2); hold on;
plot(all_tout, ya_pred_val, 'g--');
title('YawAcc (Legördülési gyorsulás) becslése'); ylabel('rad/s^2'); xlabel('Idő [s]');
grid on; legend('CarMaker (Ref)', 'PINN Predikció');

% 3. ÁBRA: Globális Trajektória 
figure('Name', 'Globális Útvonal');
plot(X_ref, Y_ref, 'b', 'LineWidth', 2); hold on;
plot(X_p, Y_p, 'r--', 'LineWidth', 1.5);
title('Jármű globális útvonalának rekonstrukciója'); 
xlabel('Globális X [m]'); ylabel('Globális Y [m]'); 
axis equal; grid on; legend('Valódi út (Ref)', 'PINN rekonstruált út');

% -------------------------------------------------------------------------
% 7. EXPORTÁLÁS SIMULINKHEZ
% -------------------------------------------------------------------------
pinn_params = struct();
pinn_params.W1 = double(extractdata(net.Learnables.Value{1})); 
pinn_params.b1 = double(extractdata(net.Learnables.Value{2})); 
pinn_params.W2 = double(extractdata(net.Learnables.Value{3})); 
pinn_params.b2 = double(extractdata(net.Learnables.Value{4})); 
pinn_params.W3 = double(extractdata(net.Learnables.Value{5})); 
pinn_params.b3 = double(extractdata(net.Learnables.Value{6})); 
pinn_params.mu = double(mu_in);
pinn_params.sigma = double(sigma_in);

save('trained_pinn_params.mat', 'pinn_params');
fprintf('KÉSZ: A paraméterek és a trajektória elemzése befejeződött.\n');

% -------------------------------------------------------------------------
% MODEL LOSS FÜGGVÉNY
% -------------------------------------------------------------------------
function [loss, gradients] = modelLoss(net, dlInputs, dlTargets, p, lambda, raw_in)
    preds = forward(net, dlInputs);
    vy_pred = preds(1, :);
    ya_pred = preds(2, :);
    
    lossData = l2loss(vy_pred, dlTargets(1, :)) + l2loss(ya_pred, dlTargets(2, :));
    
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