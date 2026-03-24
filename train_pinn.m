%% PINN_DUAL_TARGET_FINAL.m
% -------------------------------------------------------------------------
% 1. Adatkinyerés (YawAcc hozzáadva a targetekhez)
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
% Bemenetek: [delta; vx; omega; ay; ax] -> (yawacc kikerült)
inputs = [all_sim_delta, all_sim_vx, all_sim_omega, all_sim_ay, all_sim_ax]';

% Targetek: [vy; yawacc] -> (Két kimenet!)
targets = [all_sim_vy, all_sim_yawacc]';

% Normalizálás
mu_in = mean(inputs, 2); sigma_in = std(inputs, 0, 2) + 1e-6;
inputs_norm = (inputs - mu_in) ./ sigma_in;

dlInputs = dlarray(inputs_norm, 'CB');
dlTargets = dlarray(targets, 'CB');

% Pacejka paraméterek (maradnak a cftool-os értékek)
params.B = 25.046; params.C = 1.138; params.D = 8272.4;
params.m = 2108.0; params.Iz = 3954.288; params.lf = 1.47; params.lr = 1.50;

% -------------------------------------------------------------------------
% 3. Hálózat és Tanítás (Kimeneti réteg mérete most már 2)
% -------------------------------------------------------------------------
net = dlnetwork([
    featureInputLayer(size(inputs, 1))
    fullyConnectedLayer(40) % Picit növelve a kapacitás a két cél miatt
    tanhLayer
    fullyConnectedLayer(40)
    tanhLayer
    fullyConnectedLayer(2) % Kimenet: [vy; yawacc]
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
% 4. Plotolás (Két külön ábra a két kimenetnek)
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
% 5. ModelLoss (Két kimenet kezelése)
% -------------------------------------------------------------------------
function [loss, gradients] = modelLoss(net, dlInputs, dlTargets, p, lambda, raw_in)
    preds = forward(net, dlInputs);
    vy_pred = preds(1, :);
    ya_pred = preds(2, :);
    
    % Adat-alapú hiba (mindkét kimenetre)
    lossData = l2loss(vy_pred, dlTargets(1, :)) + l2loss(ya_pred, dlTargets(2, :));
    
    % Fizikai számítások
    delta = raw_in(1, :); vx = max(raw_in(2, :), 0.8); omega = raw_in(3, :); ay_meas = raw_in(4, :);
    
    af = delta - (vy_pred + (p.lf * omega)) ./ vx;
    ar = (p.lr * omega - vy_pred) ./ vx;
    
    Fyf = p.D * sin(p.C * atan(p.B * af));
    Fyr = p.D * sin(p.C * atan(p.B * ar));
    
    % Fizikai kényszer: a hálózat által becsült ya_pred-et is összevetjük a fizikai képlettel!
    lossAccel = l2loss((Fyf + Fyr) / p.m, ay_meas);
    lossMomentum = l2loss((Fyf * p.lf - Fyr * p.lr) / p.Iz, ya_pred); % A becsült yawacc-ot kényszerítjük
    
    lossPhys = 0.5 * lossAccel + 0.5 * lossMomentum;
    loss = lossData + lambda * lossPhys;
    gradients = dlgradient(loss, net.Learnables);
end