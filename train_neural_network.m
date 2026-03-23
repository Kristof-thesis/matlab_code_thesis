%% train_neural_network.m
% 1. Adatok betöltése a munkaterületről
try
    
    required_vars = {'all_sim_delta', 'all_sim_vx', 'all_sim_vy', 'all_sim_omega', 'all_sim_ay', 'all_tout'};
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
        all_sim_ay'     
    ];

    % Kimeneti adatok (Y)
    targets = all_sim_vy'; 

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
hiddenLayerSize = [8 5]; 


net = fitnet(hiddenLayerSize, 'trainbr'); 

% Tanítási paraméterek finomhangolása
net.trainParam.showWindow = true;       
net.trainParam.showCommandLine = true;  
net.trainParam.epochs = 500;           
net.trainParam.goal = 1e-5;           

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

% Rajzoljuk ki csak az időbeli lefutást (ez gyorsabb)
figure('Name', 'Laterális sebesség ellenőrzés');
plot(all_tout, targets, 'b', 'LineWidth', 1.5, 'DisplayName', 'CarMaker (Valóság)');
hold on;
plot(all_tout, outputs, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Neurális Háló');
xlabel('Idő [s]');
ylabel('Vy [m/s]');
title('A tanítás eredménye');
legend show;
grid on;