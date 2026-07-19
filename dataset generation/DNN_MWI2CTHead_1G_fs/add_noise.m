clc,
clear, close all

%%
matfiles = dir('*.mat');
for idx = 1:length(matfiles)
    filename = matfiles(idx).name;
    load(filename);
    
    % -80dB noise
    Pn_dBm=-80;
    Ez_s_m_80= addNoise_dBm(Ez_s_m, Pn_dBm);

    % -60 dB SNR
    Pn_dBm=-60;
    Ez_s_m_60= addNoise_dBm(Ez_s_m, Pn_dBm);
  

    % 40 dB SNR
    Pn_dBm=-40;
    Ez_s_m_40= addNoise_dBm(Ez_s_m, Pn_dBm);

save(filename, 'Ez_s_m_80','Ez_s_m_60','Ez_s_m_40', '-append');
disp(['New file saved for set: ' num2str(idx)])
    clearvars -except matfiles
end


function y = addNoise_dBm(x, Pn_dBm, R)
    if nargin < 3, R = 1; end                 % default 1 Ω
    Pn_W  = 10.^((Pn_dBm - 30)/10);           % dBm -> watts
    sigma2 = Pn_W * R;                        % σ^2 = V_rms^2 = P*R
    y = x + sqrt(sigma2) * randn(size(x));    % white Gaussian noise
end