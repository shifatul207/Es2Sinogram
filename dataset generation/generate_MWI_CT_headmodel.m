function [CT_output,MW_output]= generate_MWI_CT_headmodel(f, stroke_data, MW_data,  CT_data,matching)
%% inputs
isch_hem= stroke_data.isch_hem;
stroke_x= stroke_data.stroke_x;
stroke_y= stroke_data.stroke_y;
stroke_a= stroke_data.stroke_a;
stroke_b= stroke_data.stroke_b;
dim_head= stroke_data.dim_head;
antenna_rad= MW_data.antenna_rad;
no_antennas= MW_data.no_antennas;
dim_CT= CT_data.dim_CT;
theta_radon=CT_data.theta_radon;
E= CT_data.E;
%%
N_grid= 100;
profile=  imread('default_brain2.PNG');
profile= flipud(im2double(profile(:,:,3) ));
profile= smoothdata2(profile, 'movmedian',16); %% only available after matlab 2023 or later release
profile= imresize(profile,[N_grid*10+1, N_grid*10+1]);

%% assignment of stroke in profile for stroke
dx= dim_head/(N_grid*10);
x=-dim_head/2:dx:dim_head/2;
y= -dim_head/2:dx:dim_head/2;
[X,Y]= meshgrid(x,y);
x_nE= X(:);
y_nE= Y(:);

if ( ((abs(stroke_x)+stroke_a)> 0.25*dim_head) || ((abs(stroke_y)+stroke_b)> 0.3*dim_head) ) %making sure stroke is inside brain
    stroke_x= -0.01 + (0.01 + 0.01) * rand();
    stroke_y=  -0.0125 + (0.0125 + 0.0125) * rand();
    stroke_a= 0.0075 + (0.01 - 0.0075) * rand();
    stroke_b= 0.005 + (0.0125 - 0.005) * rand();
end
profile(  (  ((x_nE-stroke_x)/stroke_a).^2 + ((y_nE-stroke_y)/stroke_b).^2 )<=1  )= 0.85; % blood

%% MWT and CT contrast map

[eps, losst, HU, ~]= calculate_biological_tissue(f,E);
eps= eps([2,3,4,5,6,9]);
losst= losst([2,3,4,5,6,9]);
HU= HU([2,3,4,5,6,9]);
eps_avghead= 0.5*(eps(3)*(1-1i*losst(3))+ eps(4)*(1-1i*losst(4)));
HU_ref= 0;

profile_MWT= zeros(size(profile)); %% resize only after the contrasts are assigned
profile_MW2CT= zeros(size(profile)); %% resize after assigning contrast, zeropadding for size

profile_MWT(profile>=0.01 & profile <0.19)=eps(2)*(1-1i*losst(2)); %bone
profile_MW2CT(profile>=0.01 & profile <0.19)=HU(2); %bone
profile_MWT(profile>0.5 & profile<0.6 )=eps(4)*(1-1i*losst(4)); %white
profile_MW2CT(profile>0.5 & profile<0.6 )= HU(4); %white
profile_MWT(profile>0.45 & profile<0.6 & real(profile_MWT)~=eps(4) )= eps_avghead; %avghead
profile_MW2CT(profile>0.45 & profile<0.6 & real(profile_MWT)~=eps(4) )= HU_ref; %avghead
profile_MWT(profile>0.45 & profile <0.6 & real(profile_MWT)~=eps(4) & real(profile_MWT)~=eps_avghead)=eps(6)*(1-1i*losst(6)); %skin
profile_MW2CT(profile>0.45 & profile <0.6 & real(profile_MWT)~=eps(4) & real(profile_MWT)~=eps_avghead)=HU(6); %skin
profile_MWT(profile>0.6 & profile<0.8 )=eps(3)*(1-1i*losst(3)); % grey
profile_MW2CT(profile>0.6 & profile<0.8 )=HU(3); % grey
switch isch_hem
    case 1 %ischemic
        profile_MWT(profile==0.85)=0.9*eps(4);
        profile_MW2CT(profile==0.85)=0.995*HU(5);
        profile(profile==0.85)=128/255;  %% assigning ischemia
        
    case 2 %hemorrhagic
        profile_MWT(profile==0.85)= eps(1)*(1-1i*losst(1));
        profile_MW2CT(profile==0.85)=HU(1);
        profile(profile==0.85)=225/255; %% assigning blood
end
profile_MWT(profile>0.9)= eps(5)*(1-1i*losst(5)); % CSF
profile_MW2CT(profile>0.9)= HU(5); % CSF

if matching
    profile_MWT(abs(profile_MWT)<0.01)=matching*eps_avghead; % slack average head
else
    profile_MWT(abs(profile_MWT)<0.01)=1; % slack freespace
end
profile_MW2CT(abs(profile_MW2CT)<0.01)=HU_ref; % slack average head

%% reshaping mwi permittivity data to 101x101 grid
profile_MWT= imresize(profile_MWT,[N_grid+1, N_grid+1]);
dx_MW= dim_head/(N_grid);
x=-dim_head/2:dx_MW:dim_head/2;
y= -dim_head/2:dx_MW:dim_head/2;

%% setting appropriate dimension for CT scanning
x_CT=-dim_CT/2:dx:dim_CT/2;
y_CT= x_CT;
[X_CT,Y_CT]= meshgrid(x_CT,y_CT);
x_nE_CT= X_CT(:);
y_nE_CT= Y_CT(:);

N_grid_CT= floor(N_grid*10*dim_CT/dim_head)-mod(floor(N_grid*10*dim_CT/dim_head),2);
profile_CT= real(HU_ref)*ones(N_grid_CT+1, N_grid_CT+1);
profile_CT(N_grid_CT/2-N_grid*10/2+1:N_grid_CT/2+N_grid*10/2+1,...
    N_grid_CT/2-N_grid*10/2+1:N_grid_CT/2+N_grid*10/2+1)=profile_MW2CT;

N_grid_CT=200;
profile_CT= imresize(profile_CT,[N_grid_CT+1, N_grid_CT+1]); %normalized HU contrast

%% generate MWI data
eb= eps_avghead;
theta_deg= 360/no_antennas;
antenna_x= antenna_rad*cosd(0*theta_deg:theta_deg:(no_antennas-1)*theta_deg);
antenna_y= antenna_rad*sind(0*theta_deg:theta_deg:(no_antennas-1)*theta_deg);
antenna_coord= [antenna_x; antenna_y]';
[x_nE, y_nE, ~, ~, ~,~, ~, Ez_s_m]= generate_synthetic_Ez_data(f, eb, x, y, profile_MWT, antenna_coord );

%% generate ct sinograms
sinogram= radon(profile_CT,theta_radon);

%% MW_data
MW_output.eb=eb;
MW_output.x_nE=x_nE;
MW_output.y_nE=y_nE;
MW_output.profile_MW= profile_MWT;
MW_output.Ez_m= Ez_s_m;

CT_output.profile_CT= profile_CT;
CT_output.sinogram= sinogram;
end


