function [eps, losst, HU, tissuename]= calculate_biological_tissue(f,E)
foldername= [pwd '\Tissueproperties'];
csv_files=  dir(fullfile(foldername,'*.csv'));
for idx= 1:length(csv_files)
    filename= csv_files(idx).name;
    fileloc= [foldername '\' filename];
    filedata = readmatrix (fileloc);
    f_data= filedata(:,1);
    epsilon(:,idx)= filedata(:,2);
    losst(:,idx)= filedata(:,3);
end
eps= interp1(f_data, epsilon, f);
losst= interp1(f_data, losst, f);

%% ct scan data
foldername_CT= [pwd '\Tissueproperties_CT'];
csv_files=  dir(fullfile(foldername_CT,'*.csv'));

filename_density= csv_files(1).name; % density data
fileloc= [foldername_CT '\' filename_density];
filedata_d = readmatrix (fileloc,'Delimiter', '\t'); % avgbrain, avgbreast, blood, bone, CSF=water, fat, gland=muscle, skinsoft
density= [filedata_d(2:4) , filedata_d(1)*[1.0025,0.9975], filedata_d(5:8)]; % avgbreast, blood, bone, avgbrain, avgbrain, CSF=water, fat, gland=muscle, skinsoft, skinsoft
for idx= 2:length(csv_files)
    filename= csv_files(idx).name;
    fileloc= [foldername_CT '\' filename];
    filedata = readmatrix (fileloc);
    E_data= filedata(:,1);
    mu_d(:,idx-1)= filedata(:,2);
end
mu= [mu_d(:,2:4) mu_d(:,1) mu_d(:,1) mu_d(:,5:8) ];
mu = exp( interp1(log(E_data), log(mu), log(E), 'linear', 'extrap') ).*density;
mu_ref= mu(6);
% mu = exp( interp1(log(E_data), log(mu), log(E), 'linear', 'extrap') );
% convert mu into normalized HU
HU= round(1000*(mu/mu_ref-1))/1000;

tissuename= ["Avgbreast"; "blood"; "Bone";  "BrainGrey"; "BrainWhite"; "CSF" ; "Fat"; "Gland"; "SkinDry"];

end