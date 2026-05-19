%% train_neural_network.m
% 1. Adatok betöltése a munkaterületről
try
    
    required_vars = {'all_sim_delta', 'all_sim_vx', 'all_sim_vy', 'all_sim_omega', 'all_sim_ay', 'all_tout','all_sim_ax','all_sim_yawacc'};
    for i = 1:length(required_vars)
        if ~exist(required_vars{i}, 'var')
            error(['HIBA: A(z) ''' required_vars{i} ''' valtozo nem talalhato. Futtassa le a generate_training_data.m-et!']);
        end
    end
    

    tout_proc = all_tout;

    % Bemeneti adatok (X), csak a mért jelek 
    inputs = [
        all_sim_delta'; 
        all_sim_vx';    
        all_sim_omega'; 
        all_sim_ax';
        all_sim_ay'     
    ];

    % Kimeneti adatok (Y)
    targets = [all_sim_vy';
               all_sim_yawacc'];

    numSamples = size(inputs, 2);
    if numSamples == 0
        error('HIBA: Nincs minta az adatokban!');
    end

    disp(['Sikeresen betoltve ' num2str(numSamples) ' minta a CarMakerbol.']);

catch ME
    disp(['Hiba tortent az adatok betoltese soran: ' ME.message]);
    return;
end

%% 2. Neurális háló létrehozása és konfigurálása
hiddenLayerSize = [15 10]; 


net = fitnet(hiddenLayerSize, 'trainbr'); 

% Tanítási paraméterek finomhangolása
net.trainParam.showWindow = true;       
net.trainParam.showCommandLine = true;  
net.trainParam.epochs = 500;           
net.trainParam.goal = 1e-6;           

% Bayesian Regularization esetén a divideParam-ot a hálózat maga kezeli belsőleg,

net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

% Extra simítás: Adatfeldolgozás beállítása 
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};

%% 3. Neurális háló tanítása
disp('Neurális háló tanítása elindult...');
[net, tr] = train(net, inputs, targets);
disp('Neurális háló tanítása befejezve.');

save('net_carmaker_trained.mat', 'net');

%% Eredmények megjelenítése 
outputs = net(inputs); 
vy_pred = outputs(1, :);
ya_pred = outputs(2, :);

%% 4. Trajektória rekonstrukció (XY sík)
% Predikciók kinyerése
nn_out = net(inputs); 
vy_nn = nn_out(1,:); 
ya_nn = nn_out(2,:);
Ts = mean(diff(all_tout));
numS = length(all_tout);

% Inicializálás
X_nn = zeros(numS,1); Y_nn = zeros(numS,1); Psi_nn = zeros(numS,1); om_nn = zeros(numS,1);
X_ref = zeros(numS,1); Y_ref = zeros(numS,1); Psi_ref = zeros(numS,1);
om_nn(1) = all_sim_omega(1); % Kezdő szögsebesség

for t = 1:numS-1
    % NN alapú integrálás 
    om_nn(t+1)  = om_nn(t) + ya_nn(t) * Ts;
    Psi_nn(t+1) = Psi_nn(t) + om_nn(t) * Ts;
    X_nn(t+1)   = X_nn(t) + (all_sim_vx(t)*cos(Psi_nn(t)) - vy_nn(t)*sin(Psi_nn(t))) * Ts;
    Y_nn(t+1)   = Y_nn(t) + (all_sim_vx(t)*sin(Psi_nn(t)) + vy_nn(t)*cos(Psi_nn(t))) * Ts;
    
    % CarMaker referencia integrálás 
    Psi_ref(t+1) = Psi_ref(t) + all_sim_omega(t) * Ts;
    X_ref(t+1)   = X_ref(t) + (all_sim_vx(t)*cos(Psi_ref(t)) - all_sim_vy(t)*sin(Psi_ref(t))) * Ts;
    Y_ref(t+1)   = Y_ref(t) + (all_sim_vx(t)*sin(Psi_ref(t)) + all_sim_vy(t)*cos(Psi_ref(t))) * Ts;
end

% Megjelenítés
figure('Name', 'NN Trajektória Validáció');
plot(X_ref, Y_ref, 'b', 'LineWidth', 2, 'DisplayName', 'CarMaker (Ref)'); hold on;
plot(X_nn, Y_nn, 'r--', 'LineWidth', 1.5, 'DisplayName', 'NN Rekonstrukció');
title('Jármű útvonala az XY síkon (Tanított NN alapján)');
xlabel('X [m]'); ylabel('Y [m]'); axis equal; grid on; legend show;
% Első subplot: Vy
figure
plot(all_tout, targets(1,:), 'b', 'LineWidth', 1.8, 'DisplayName', 'CarMaker (Ref)');
hold on;
plot(all_tout, vy_pred, 'r--', 'LineWidth', 1.2, 'DisplayName', 'NN Predikció');
ylabel('Vy [m/s]');
title('Laterális sebesség becslése');
legend show; grid on;

% Második subplot: YawAcc
figure
plot(all_tout, targets(2,:), 'b', 'LineWidth', 1.8, 'DisplayName', 'CarMaker (Ref)');
hold on;
plot(all_tout, ya_pred, 'g--', 'LineWidth', 1.2, 'DisplayName', 'NN Predikció');
xlabel('Idő [s]');
ylabel('YawAcc [rad/s^2]');
title('Legördülési gyorsulás becslése');
legend show; grid on;

% =========================================================================
% EGYSZERŰSÍTETT MENTÉS AZ MPC SZÁMÁRA (Tiszta statisztikai számokkal)
% =========================================================================

nn_net = net; % Átnevezzük a hálót, hogy ne keveredjen a PINN-nel

% Kiszámoljuk a bemeneti adatmátrixból soronként az átlagot és a szórást
% Így pontosan 5 darab számunk lesz mindkét vektorban, amit az MPC használni tud
mu_in_nn = mean(inputs, 2);  
sigma_in_nn = std(inputs, 0, 2); 

% Védelem: Ha valamelyik szórás 0 lenne (konstans sor), ne oszthassunk nullával az MPC-ben
sigma_in_nn(sigma_in_nn == 0) = 1e-9; 

% Elmentjük a fájlt, amit az mpc_controller_classic_nn be fog tölteni
save('trained_classic_nn_data.mat', 'nn_net', 'mu_in_nn', 'sigma_in_nn');

disp('------------------------------------------------------------');
disp('A klasszikus hálózat és a normalizációs adatok elmentve!');
disp('Fájl neve: trained_classic_nn_data.mat');
disp('------------------------------------------------------------');