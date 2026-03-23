%% PINN_STIFFNESS_CALC_INTEGRATED.m
% -------------------------------------------------------------------------
% 1. Adatkinyerés
% -------------------------------------------------------------------------
if ~exist('out_vx', 'var')
    error('HIBA: Nem található az out_vx változó! Indítsa el a szimulációt.');
end

all_tout       = out_vx.Time;
all_sim_vx     = out_vx.Data;
all_sim_vy     = out_vy.Data;
all_sim_omega  = out_omega.Data;
all_sim_ay     = out_ay.Data;
all_sim_delta  = out_delta.Data;
all_sim_FyF    = out_FyF.Data;
all_sim_FyR    = out_FyR.Data;
all_sim_slipF  = out_slipF.Data;
all_sim_slipR  = out_slipR.Data;

% -------------------------------------------------------------------------
% 2. Cf és Cr SZÁMÍTÁSA
% -------------------------------------------------------------------------
validF = abs(all_sim_slipF) > 0.005; 
validR = abs(all_sim_slipR) > 0.005;

Cf_calc = abs(mean(all_sim_FyF(validF) ./ all_sim_slipF(validF)));
Cr_calc = abs(mean(all_sim_FyR(validR) ./ all_sim_slipR(validR)));

if isnan(Cf_calc) || isinf(Cf_calc), Cf_calc = 150000; end
if isnan(Cr_calc) || isinf(Cr_calc), Cr_calc = 160000; end

fprintf('Számított kanyarmerevségek:\n Cf: %.2f N/rad\n Cr: %.2f N/rad\n', Cf_calc, Cr_calc);

% -------------------------------------------------------------------------
% 3. PINN Konfiguráció
% -------------------------------------------------------------------------
inputs = [all_sim_delta'; all_sim_vx'; all_sim_omega'; all_sim_ay'; all_sim_yawacc'; all_sim_ax'];
targets = all_sim_vy';

params.m  = 2108.0;      
params.Iz = 3954.288;    
params.lf = 1.47;        
params.lr = 1.50;        
params.Cf = Cf_calc;    
params.Cr = Cr_calc;    

dlInputs = dlarray(inputs, 'CB');
dlTargets = dlarray(targets, 'CB');

% Hálózat - picit több neuron a jobb illeszkedésért
net = dlnetwork([
    featureInputLayer(size(inputs, 1))
    fullyConnectedLayer(35)
    tanhLayer
    fullyConnectedLayer(35)
    tanhLayer
    fullyConnectedLayer(size(targets, 1))
]);

% Tanítási paraméterek
numEpochs = 2000;      
learningRate = 0.002;  
lambda = 0.1;          

velocity = [];
squaredGradient = [];
lossHistory = zeros(numEpochs, 1);

% -------------------------------------------------------------------------
% 4. Tanítás (KIÍRÁSSAL)
% -------------------------------------------------------------------------
disp('PINN tanítása folyamatban...');

tic; % Időmérés indítása
for epoch = 1:numEpochs
    [loss, gradients] = dlfeval(@modelLoss, net, dlInputs, dlTargets, params, lambda);
    [net, velocity, squaredGradient] = adamupdate(net, gradients, velocity, squaredGradient, epoch, learningRate);
    
    lossVal = extractdata(loss);
    lossHistory(epoch) = lossVal;
    
    
    if mod(epoch, 100) == 0 || epoch == 1
        fprintf('Epoch %4d / %d | Loss: %.8f\n', epoch, numEpochs, lossVal);
    end
end
toc; % Időmérés vége

% -------------------------------------------------------------------------
% 5. Plotolás
% -------------------------------------------------------------------------
vy_pred = extractdata(forward(net, dlInputs));

figure('Name', 'PINN Végeredmény Stabil');
plot(all_tout, targets, 'b', 'LineWidth', 1.8, 'DisplayName', 'CarMaker');
hold on;
plot(all_tout, vy_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'PINN');
title(sprintf('Vy becslés (lambda=%.1f, Cf=%.0f)', lambda, Cf_calc));
xlabel('Idő [s]'); ylabel('m/s');
legend; grid on;

% -------------------------------------------------------------------------
% 6. ModelLoss
% -------------------------------------------------------------------------
function [loss, gradients] = modelLoss(net, dlInputs, dlTargets, p, lambda)
    vy_pred = forward(net, dlInputs);
    lossData = l2loss(vy_pred, dlTargets);
    
    % Indexek: 1:delta, 2:vx, 3:omega, 4:ay, 5:yawacc
    delta    = dlInputs(1, :);
    vx       = max(dlInputs(2, :), 0.5);
    omega    = dlInputs(3, :);
    ay_meas  = dlInputs(4, :);
    yacc_meas = dlInputs(5, :);
    
    % Szlipek és Erők
    alfa_f = delta - (vy_pred + (p.lf * omega)) ./ vx;
    alfa_r = (p.lr * omega - vy_pred) ./ vx;
    Fyf_p = p.Cf * alfa_f;
    Fyr_p = p.Cr * alfa_r;
    
    % 1. Laterális gyorsulás hiba
    lossAccel = l2loss((Fyf_p + Fyr_p) / p.m, ay_meas);
    
    % 2. Nyomaték hiba SKÁLÁZVA 
    lossMomentum = l2loss((Fyf_p * p.lf - Fyr_p * p.lr) / p.Iz, yacc_meas);
    
    % Összesített fizikai hiba 
    lossPhysics = 0.5 * lossAccel + 0.5 * lossMomentum;
    
    loss = lossData + lambda * lossPhysics;
    gradients = dlgradient(loss, net.Learnables);
end