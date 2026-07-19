clc,
clear, close all

%%
matching=1;
if matching==1
    savefolderName = 'DNN_MWI2CTHead_1G_matched';
else
    savefolderName = 'DNN_MWI2CTHead_1G_FS';
end
if ~exist(savefolderName, 'dir')
    mkdir(savefolderName);
end
%%
f=1e9;
E=1e-1;
MW_data.antenna_rad=0.25;
MW_data.no_antennas=12;
CT_data.dim_CT=0.3;
CT_data.theta_radon=0:0.5:180;
CT_data.E=E;

%%
dim_head= 0.2; % 0.2, 0.225
stroke_xs= -0.2*dim_head:0.01:0.2*dim_head;
stroke_ys= -0.25*dim_head:0.01:0.25*dim_head;
stroke_as= 0.01:0.005:0.025;
stroke_bs= 0.01:0.005:0.025;
stroke_b= stroke_bs(1);
isch_hems=[1,2];

%%
fileidx=0;
stroke_data.dim_head=dim_head;
datacount=1;
for stroke_x=stroke_xs
    for stroke_y=stroke_ys
        for stroke_a= stroke_as
            % for stroke_b=stroke_bs
            for isch_hem=isch_hems
                stroke_data.isch_hem=isch_hem;
                stroke_data.stroke_x=stroke_x;
                stroke_data.stroke_y=stroke_y;
                stroke_data.stroke_a=stroke_a;
                stroke_data.stroke_b=stroke_b;
                
                tic
                [CT_output,MW_output]= generate_MWI_CT_headmodel(f, stroke_data, MW_data,  CT_data, matching);

                eb= MW_output.eb;
                profile_MW(:,:,datacount)=MW_output.profile_MW;
                Ez_s_m(:,datacount)= MW_output.Ez_m;
                profile_CT(:,:,datacount)=CT_output.profile_CT;
                sinogram_2D(:,:,datacount)=CT_output.sinogram;
                
                disp(['MWI and CT data genereted for sample ' num2str(datacount),', time taken= ' num2str(toc) 's'])
                datacount=datacount+1;
                if datacount>2000
                    savefileName= ['MWI2CT_data_dim' num2str(dim_head*100) 'cm_b' num2str(stroke_b*100) 'cm_' num2str(fileidx) '.mat']
                    save( [savefolderName '\' savefileName],'f', 'eb','MW_output', 'Ez_s_m' ,'sinogram_2D', 'profile_CT','-v7.3'  )
                    disp(['Data saved for batch ' num2str(fileidx)]);
                    fileidx= fileidx+1;
                    datacount=1;
                    save([savefolderName '\' savefileName],'f', 'eb','profile_MW', 'profile_CT','Ez_s_m' ,'sinogram_2D', '-v7.3'  )
                end
                % end
            end
        end
    end
end
%%
savefileName= ['MWI2CT_data_dim' num2str(dim_head*100) 'cm_b' num2str(stroke_b*100) 'cm_' num2str(fileidx) '.mat']
save([savefolderName '\' savefileName],'f', 'eb','profile_MW', 'profile_CT', 'Ez_s_m' ,'sinogram_2D', '-v7.3'  )
disp(['Data saved for batch ' num2str(fileidx)]);
clear Ez_s_m sinogram_2D profile_CT MW_output
%%
disp('All data generation completed ')



