clc
clear, close all

%% loading appropriate test data, ischemic or hemorrhagic, CM or FS, noise level

srctype= 'matched'; % fs, matched
noise= num2str(60); % choose the string corresponding to the noise P_n
noise= 'clean';

isch_hem=2; % flag for choosing ischemic or hemorrhagic stroke examples, 1= ischemic, 2= hemorrhagic
if isch_hem==1 %ischemic
    dim=22.5;  
    sampleidx=12;
    b=1.5;
    a=1.0;
    xc= 0.01;
    yc=-0.007;
    [xx, yy] = generate_ellipse(1*1e-2, 1.25*1e-2, xc, yc);
    stroke_str= 'isch';
    colormarker= 'w-.';
elseif isch_hem==2
    %hemorr: 20, 115, 2
    dim=20;  
    sampleidx=115;
    b=2;
    a=1.5;
    xc= -0.031;
    yc=-0.02;
    [xx, yy] = generate_ellipse(1.5*1e-2, 2*1e-2, xc, yc);
    stroke_str= 'hem';
    colormarker= 'k-.';
else
    error('gimme stroke')
end

%%
fs=20; %fontsize
window= [-50, 100]; % window size

%% load no stroke data
filename_nostroke= ['MWI2CT_data_dim' num2str(dim) 'cm_nostroke_' srctype '.mat'];
data_nostroke= load(['test_stroke_data\' filename_nostroke]);
eb= data_nostroke.eb;
E_s_nostroke= data_nostroke.Ez_s_m;
MW_contrast_nostroke= data_nostroke.MW_output.profile_MW/eb-1;
CT_profile_nostroke= data_nostroke.profile_CT;
CT_profile_nostroke_windowed= windowed_CT(data_nostroke.profile_CT,window);


%% load stroke data
filename_stroke= ['test_stroke_data\MWI2CT_data_dim' num2str(dim) 'cm_b' num2str(b) 'cm_0' srctype '_' stroke_str];
data_stroke= load( [filename_stroke '.mat']);
E_s_stroke= data_stroke.E_s_m;
MW_contrast= data_stroke.MW_contrast;
CT_profile= data_stroke.CT_profile;
CT_profile_windowed= windowed_CT(CT_profile,window);
E_s_diff= E_s_stroke- E_s_nostroke;

%% taking MW output for MW imaging
MW_data= data_nostroke.MW_output;
x_nE= MW_data.x_nE;
y_nE= MW_data.y_nE;
eb= MW_data.eb;
f=1e9;
antenna_rad=0.25; % antenna radius, considered when generating the dataset
no_antennas=12; % number of antennas, considered when generating the dataset

theta_deg= 360/no_antennas;
antenna_x= antenna_rad*cosd(0*theta_deg:theta_deg:(no_antennas-1)*theta_deg);
antenna_y= antenna_rad*sind(0*theta_deg:theta_deg:(no_antennas-1)*theta_deg);
antenna_coord= [antenna_x; antenna_y]';
%% Compute the forward solution to extract the greens function, reference fields
disp ('Computing the forward solution E field data')
[~, ~, GD, GS, Ez_inc_o, ~, Ez_o, ~]= ...
    generate_synthetic_Ez_data(f, eb, unique(x_nE), unique(x_nE),...
    data_nostroke.MW_output.profile_MW, antenna_coord );
masking= sqrt( x_nE.^2 + y_nE.^2) >= dim*1e-2/2;
dx= x_nE(2)- x_nE(1);
target_dim= 0.15+dx;
N=100;
dim_hem=0.25;
dim_isch= 0.20;
dx_hem= dim_hem/N;
x_nE_hem= -dim_hem/2:dx_hem:dim_hem/2; y_nE_hem= x_nE_hem;
dx_isch= dim_isch/N;
x_nE_isch= -dim_isch/2:dx_isch:dim_isch/2; y_nE_isch= x_nE_isch;

%% CT scan profile dimensions
N_CT= 201;
dx_CT= 0.3/(N_CT-1);
x_nE_CT= -0.15:dx_CT: 0.15;
y_nE_CT= x_nE_CT;

%% loading the CT data from predictions
CT_filename= [srctype '_' noise 'dim' num2str(dim) '_sampleidx' num2str(sampleidx) 'b' num2str(b) '.mat' ];
CT_data= load(['test_model_outputs\' CT_filename]);
theta_radon=0:0.5:180;

CT_pred= fliplr(CT_data.pred_img);
CT_pred_windowed= flipud(windowed_CT(CT_pred(1:201,1:201),window));

CT_diff_pred= iradon(CT_data.pred_sino-data_nostroke.sinogram_2D, theta_radon);
CT_diff_pred_windowed= windowed_CT(CT_diff_pred(1:201, 1:201),window);

%% _______________ absolute imaging _____________________

%% parameters for the BIM algorithms
params.eb= eb;
params.f= f;
params.alpha= 1;
params.LW_iter= 10;
params.GS= GS;
params.GD= GD.';
params.niter=5;
if strcmp(srctype, 'fs') % force regularize to get a convergent solution
    params.alpha= 0.5;
    params.LW_iter= 5;
end
params.masking= masking;

fields.E_ref_o= Ez_inc_o.';
fields.E_s= E_s_stroke;

[delta_eps_abs, ~,~] = BIM_LW_TM (params, fields);
N= length(delta_eps_abs);
delta_eps_abs= reshape(delta_eps_abs, sqrt(N), sqrt(N) );
% delta_eps_abs= smoothdata2 (delta_eps_abs, 'sgolay',8);
%% _______________differential imaging ____________________
fields.E_s= E_s_diff;
[delta_eps_diff, ~,~] = BIM_LW_TM (params, fields);
delta_eps_diff= reshape(delta_eps_diff*eb, sqrt(N), sqrt(N) );
% delta_eps_diff= smoothdata2 (delta_eps_diff, 'sgolay',8);
%% _s represents subscript for the differential data, independent of the demo strokes
dim_s=dim/100; % conversion to cm
dx_s= dim_s/100; % grid
x_nE_s= -dim_s/2:dx_s:dim_s/2; y_nE_s= x_nE_s;

figure
image(x_nE_s,y_nE_s,real(MW_contrast),'CDataMapping','scaled')
xlabel('x(m)')
ylabel('y(m)')
grid on;
axis equal
axis tight
set(gca,'YDir','normal')
cb=colorbar('southoutside'); set(gca, 'fontsize', fs)
xticks([-0.1 0 0.1]);
yticks([-0.1 0 0.1]);
set(gcf, 'Position', [100,100, 500,500])
caxis([-0.75 0.75]);  
cb.Ticks = linspace(-0.75, 0.75, 3);  
colormap gray

[x_s_new, y_s_new, delta_eps_abs_new]= zero_pad_images(x_nE_s.', y_nE_s.',delta_eps_abs, target_dim);
[~, ~, delta_eps_diff_new]= zero_pad_images(x_nE_s.', y_nE_s.',delta_eps_diff, target_dim);

figure
image(x_s_new,y_s_new, real(delta_eps_abs_new), 'CDatamapping', 'scaled')
xlabel('x(m)')
ylabel('y(m)')
grid on;
axis equal
axis tight
set(gca,'YDir','normal')
% title({'Reconstructed', '(Absolute)' })
cb=colorbar('southoutside'), set(gca, 'fontsize', fs)
xticks([-0.15 0 0.15]);
yticks([-0.15 0 0.15]);
set(gcf, 'Position', [100,100, 500,500])
caxis([-0.25 0.25]);  
cb.Ticks = linspace(-0.25, 0.25, 3);  
colormap gray

figure
image(x_s_new, y_s_new, real(delta_eps_diff_new), 'CDatamapping', 'scaled')
hold on
plot (xx,yy, colormarker, 'linewidth',2)
xlabel ('pixel index (x)')
ylabel ('pixel index (y)')
grid on;
axis equal
axis tight
set(gca,'YDir','normal')
% title({'Reconstructed', '(Differential)' })
cb=colorbar('southoutside'); set(gca, 'fontsize', fs);
xticks([-0.15 0 0.15]);
yticks([-0.15 0 0.15]);
set(gcf, 'Position', [100,100, 500,500]);
if isch_hem==1
caxis([-0.25 0.25]);  
cb.Ticks = linspace(-0.25, 0.25, 3); 
else
caxis([-1.25 1.25]);  
cb.Ticks = linspace(-1.25, 1.25, 3); 
end
colormap gray

%% plotting images (CT)
figure
image(1:201, 1:201, CT_profile_windowed,'CDataMapping','scaled')
% xlabel('x(m)')
% ylabel('y(m)')
grid on;
axis equal
axis tight
set(gca,'YDir','normal')
% title({'True', 'Image'})
cb=colorbar('southoutside'); set(gca, 'fontsize', fs)
xticks([1 50 100 150 201]);
yticks([1 50 100 150 201]);
set(gcf, 'Position', [100,100, 500,500])
caxis([-50 100]);  
cb.Ticks = linspace(-50, 100, 3);  
colormap gray

figure
image(x_nE_CT, y_nE_CT, CT_pred_windowed, 'CDatamapping', 'scaled')
% xlabel('x(m)')
% ylabel('y(m)')
grid on;
axis equal
axis tight
set(gca,'YDir','normal')
% title({'Reconstructed', '(Absolute)' })
cb=colorbar('southoutside'); set(gca, 'fontsize', fs)
xticks([-0.15 0 0.15]);
yticks([-0.15 0 0.15]);
set(gcf, 'Position', [100,100, 500,500])
caxis([-50 100]);  
cb.Ticks = linspace(-50, 100, 3);  
colormap gray

figure
image(x_nE_CT, y_nE_CT, CT_diff_pred_windowed, 'CDatamapping', 'scaled')
hold on
plot (xx,yy, colormarker, 'linewidth',2)
% xlabel('x(m)')
% ylabel('y(m)')
grid on;
axis equal
axis tight
set(gca,'YDir','normal')
% title({'Reconstructed', '(Differential)' })
cb=colorbar('southoutside'); set(gca, 'fontsize', fs)
xticks([-0.15 0 0.15]);
yticks([-0.15 0 0.15]);
set(gcf, 'Position', [100,100, 500,500])
caxis([-30 30]);  
cb.Ticks = linspace(-30, 30, 3);  
colormap gray


%% Compute normalized cross correlation
ncc_MW_abs = corr2(real(MW_contrast), real(delta_eps_abs))
ncc_MW_diff = corr2(real(MW_contrast-MW_contrast_nostroke), real(delta_eps_diff))
ncc_CT_abs = corr2(CT_profile_windowed, CT_pred_windowed)
ncc_CT_diff= corr2(CT_profile_windowed- CT_profile_nostroke_windowed, CT_diff_pred_windowed)
%% Compute Gradient Magnitude correlation
gmc_MW_abs = GMC(real(MW_contrast), real(delta_eps_abs))
gmc_MW_diff = GMC(real(MW_contrast-MW_contrast_nostroke), real(delta_eps_diff))
gmc_CT_abs = GMC(CT_profile_windowed, CT_pred_windowed)
gmc_CT_diff= GMC(CT_profile_windowed- CT_profile_nostroke_windowed, CT_diff_pred_windowed)

%% plotting absolute groundn truth
MW_abs_true= real(MW_contrast);
CT_abs_true= CT_profile_windowed;
[~, ~, MW_abs_true_new]= zero_pad_images(x_nE_s.', y_nE_s.', MW_abs_true, target_dim);

figure
image(x_s_new,y_s_new,real(MW_abs_true_new),'CDataMapping','scaled')
% xlabel('x(m)'), ylabel('y(m)')
grid on; axis equal; axis tight; set(gca,'YDir','normal')
cb=colorbar('southoutside'); set(gca, 'fontsize', fs)
xticks([-0.15 0 0.15]);
yticks([-0.15 0 0.15]);
set(gcf, 'Position', [100,100, 500,500])
caxis([-0.75 0.75]);  
cb.Ticks = linspace(-0.75, 0.75, 3);  
colormap gray

figure
image(x_nE_CT,y_nE_CT,CT_abs_true, 'CDatamapping', 'scaled')
% xlabel('x(m)'), ylabel('y(m)')
grid on; axis equal; axis tight; set(gca,'YDir','normal')
cb=colorbar('eastoutside'); set(gca, 'fontsize', fs)
xticks([-0.15 0 0.15]);
yticks([-0.15 0 0.15]);
set(gcf, 'Position', [100,100, 500,500])
caxis([-50 100]);  
cb.Ticks = linspace(-50, 100, 3);  
colormap gray


%% plotting differntial groundn truth
MW_diff_true= real(MW_contrast-MW_contrast_nostroke);
CT_diff_true= CT_profile_windowed- CT_profile_nostroke_windowed;

[~, ~, MW_diff_true_new]= zero_pad_images(x_nE_s.', y_nE_s.', MW_diff_true, target_dim);

figure
image(x_s_new,y_s_new,real(MW_diff_true_new),'CDataMapping','scaled')
% hold on
% plot (xx,yy, 'k:', 'linewidth',2)
xlabel('x(m)'), ylabel('y(m)')
grid on; axis equal; axis tight; set(gca,'YDir','normal')
% title({'MW (differential)'})
cb=colorbar('southoutside'); set(gca, 'fontsize', fs)
xticks([-0.15 0 0.15]);
yticks([-0.15 0 0.15]);
set(gcf, 'Position', [100,100, 500,500])
caxis([-0.25 0.25]);  
cb.Ticks = linspace(-0.25, 0.25, 3);  
colormap gray

figure
image(x_nE_CT, y_nE_CT,CT_diff_true, 'CDatamapping', 'scaled')
% hold on
% plot (xx,yy, 'k:', 'linewidth',2)
xlabel('x(m)'), ylabel('y(m)')
grid on; axis equal; axis tight; set(gca,'YDir','normal')
cb=colorbar('eastoutside'); set(gca, 'fontsize', fs)
xticks([-0.15 0 0.15]);
yticks([-0.15 0 0.15]);
set(gcf, 'Position', [100,100, 500,500])
caxis([-30 30]);  
cb.Ticks = linspace(-30, 30, 3);  
colormap gray


%% function 
function [x_new_flat, y_new_flat, im_new_2D] = zero_pad_images(x_nE, y_nE, im_old, target_dim) 
    % target_dim should be > max(x_nE) to add padding
    
    % 1. Extract unique 1D axes from the flattened grid inputs
    x_axis = unique(x_nE)'; 
    y_axis = unique(y_nE)';
    dx = x_axis(2) - x_axis(1);
    
    % 2. Calculate the extended 1D axes
    pad_left_x  = fliplr( x_axis(1)-dx : -dx : -target_dim );
    pad_right_x = x_axis(end)+dx : dx : target_dim;
    x_axis_new = [pad_left_x, x_axis, pad_right_x];
    
    pad_left_y  = fliplr( y_axis(1)-dx : -dx : -target_dim );
    pad_right_y = y_axis(end)+dx : dx : target_dim;
    y_axis_new = [pad_left_y, y_axis, pad_right_y];
    
    % 3. Create the new padded 2D matrices
    [X_new, Y_new] = ndgrid(x_axis_new, y_axis_new);
    im_new_2D = zeros(size(X_new)); 
    
    % 4. Calculate exact placement indices
    idx_x = (1:length(x_axis)) + length(pad_left_x);
    idx_y = (1:length(y_axis)) + length(pad_left_y);
    
    % 5. Drop the old image exactly into the allocated indices
    im_new_2D(idx_x, idx_y) = im_old;
    
    % 6. Flatten spatial coordinates to match the original x_nE / y_nE format
    x_new_flat = X_new(:);
    y_new_flat = Y_new(:);
end


function y= windowed_CT(CT_image, window)
    y= CT_image*1000;
    y(y<window(1))= window(1);
    y(y>window(2))= window(2);

end

function y= GMC(img_true, img_rec)
[Gx1, Gy1] = gradient(img_true);
[Gx2, Gy2] = gradient(img_rec);

grad_true = sqrt(Gx1.^2 + Gy1.^2);
grad_rec  = sqrt(Gx2.^2 + Gy2.^2);

y = corr2(grad_true, grad_rec);
end

function [x, y] = generate_ellipse(a, b, xc, yc)
    if a > b
        eccentricity = sqrt(1 - b^2/a^2);
    else
        eccentricity = sqrt(1 - a^2/b^2);
    end
    
    numPoints = 100;
    
    t = linspace(0, 2 * pi, numPoints);

    x = xc + a * cos(t);
    y = yc + b * sin(t);
end
