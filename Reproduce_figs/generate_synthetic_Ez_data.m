function [x_nE, y_nE, GD, GS, Ez_inc_o, Ez_inc_m, Ez_o, Ez_s_m]= generate_synthetic_Ez_data(f, eb, x, y, profile, antenna_coord )
e0= 8.854e-12;
u0= pi*4e-7;
const= -2*pi*f*u0/4;
dx= x(2)- x(1);
an= sqrt(dx^2/pi);
lda= 3e8/f;
kb= 2*pi*sqrt(eb)/lda;
%% creating the source and receiver coordinates assuming multiillumination approach
[no_antennas, ~]= size(antenna_coord);
src_coord=[];
rec_coord= [];
for i= 1: no_antennas
    for j= i+1:no_antennas
        src_coord= [src_coord; antenna_coord(i,:)];
        rec_coord= [rec_coord; antenna_coord(j,:)];
    end
end
M= no_antennas* (no_antennas-1)/2;
[X,Y]= ndgrid(x,y);
x_nE= X(:);
y_nE= Y(:);
N= length(x_nE);
%% creating the Greens function for the state equation
diff_x= repmat(x_nE,1,N)- repmat(x_nE',N,1);
diff_y= repmat(y_nE,1,N)- repmat(y_nE',N,1);
[~,rho_GS] = cart2pol(diff_x,diff_y);
GS= G2D (kb, rho_GS,an);

%% constructing the Greens function for the data equation
diff_recx= repmat(rec_coord(:,1),1,N)- repmat(x_nE',M,1);
diff_recy= repmat(rec_coord(:,2),1,N)- repmat(y_nE',M,1);
[~,rho_GD] = cart2pol(diff_recx,diff_recy);
GD=  G2D (kb, rho_GD, an);

%% Computing Ez_inc_o
diff_srcx= repmat(antenna_coord(:,1),1,N)- repmat(x_nE',no_antennas,1);
diff_srcy= repmat(antenna_coord(:,2),1,N)- repmat(y_nE',no_antennas,1);
[~,rho]=cart2pol(diff_srcx,diff_srcy);
Ez_inc_o= const*besselh(0,2,kb*rho);
Ez_inc_o(rho<dx)=0;

%% Computing Ez_inc_m using greens functions
diff_src_recx= src_coord(:,1)- rec_coord(:,1);
diff_src_recy= src_coord(:,2)- rec_coord(:,2);
[~,rho_txrx]=cart2pol(diff_src_recx,diff_src_recy);
Ez_inc_m= const*besselh(0,2,kb*rho_txrx);
Ez_inc_m(rho<dx)=0;

%% create measurement copies for incident fields
src_idx= create_seq(no_antennas);
Ez_inc_o_full=[];
for antenna_idx= 1:no_antennas
    Ez_inc_o_full= [Ez_inc_o_full; repmat(Ez_inc_o(antenna_idx,:),no_antennas-antenna_idx,1)];
end

%% Computing the total field at the object and scattered field at the receivers
contrast= profile(:)/eb-1;
Ez_o= Ez_inc_o/transpose(eye(N)+GS*diag(contrast)); % updating the object field using state equation

Ez_o_full=[];
for antenna_idx= 1:no_antennas
    Ez_o_full= [Ez_o_full; repmat(Ez_o(antenna_idx,:),no_antennas-antenna_idx,1)];
end
Ez_s_m=  (-GD.*(Ez_o_full))*(contrast); % updating the scattered field using Data equation
end


function y= G2D (kb,rho,an)
y= zeros(size(rho));
y(rho==0)=1j/2* (pi*kb*an*besselh(1,2,kb*an)-2j);
y(rho~=0)= 1j/2*pi*kb*an*besselj(1,kb*an)*besselh(0,2,kb*rho(rho~=0));
end

function y= create_seq(N)
delta= N-1;
y=[1];
while(delta>1)
    idx= y(end)+delta;
    y= [y idx];
    delta= delta-1;
end
end