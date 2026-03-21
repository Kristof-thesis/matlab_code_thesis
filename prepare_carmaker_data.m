%% generate_training_data.m (Minden jellel)
disp('CarMaker adatok kinyerése a Workspace-ből...');

% Ellenőrzés
if ~exist('out_vx', 'var')
    error('HIBA: Nem található az out_vx változó!');
end

% 1. Idővektor
all_tout = out_vx.Time;

% 2. Alap adatok
all_sim_vx    = out_vx.Data;
all_sim_vy    = out_vy.Data;
all_sim_omega = out_omega.Data;
all_sim_ay    = out_ay.Data;
all_sim_delta = out_delta.Data;
all_sim_ax    = out_ax.Data;
% 3. Erők (A Newton-egyenlethez)
all_sim_FyF   = out_FyF.Data;
all_sim_FyR   = out_FyR.Data;

% 4. Slip szögek (A gumi-dinamika ellenőrzéséhez)
all_sim_slipF = out_slipF.Data;
all_sim_slipR = out_slipR.Data;

% 5. Gyorsulás (Az inercia-egyenlethez)
all_sim_yawacc = out_yawacc.Data; 

numSamples = length(all_tout);
fprintf('Sikeresen betöltve %d minta (Minden jel a helyén!).\n', numSamples);