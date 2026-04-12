%% generate_training_data.m (Ritkított adatokkal: 1ms -> 20ms)
disp('CarMaker adatok kinyerése és ritkítása...');

% Ellenőrzés
if ~exist('out_vx', 'var')
    error('HIBA: Nem található az out_vx változó!');
end

% ---  PARAMÉTEREK ---
step = 10; 
indices = 1:step:length(out_vx.Time);

% 1. Idővektor 
all_tout = out_vx.Time(indices);

% 2. Alap adatok 
all_sim_vx    = out_vx.Data(indices);
all_sim_vy    = out_vy.Data(indices);
all_sim_omega = out_omega.Data(indices);
all_sim_ay    = out_ay.Data(indices);
all_sim_delta = out_delta.Data(indices);
all_sim_ax    = out_ax.Data(indices);

% 3. Erők 
all_sim_FyF   = out_FyF.Data(indices);
all_sim_FyR   = out_FyR.Data(indices);

% 4. Slip szögek
all_sim_slipF = out_slipF.Data(indices);
all_sim_slipR = out_slipR.Data(indices);

% 5. Gyorsulás 
all_sim_yawacc = out_yawacc.Data(indices); 

numSamples = length(all_tout);
new_Ts = mean(diff(all_tout)); 

fprintf('Sikeresen kinyerve %d minta (Ritkítva minden %d. adat).\n', numSamples, step);
fprintf('Az új mintavételi idő (Ts): %.3f másodperc.\n', new_Ts);