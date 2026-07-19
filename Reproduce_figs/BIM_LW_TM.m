function [contrast, fields_out,params_out] = BIM_LW_TM (params, fields)
% ___________________Description______________________
% params
    % eb= background complex dielectric constant = a-jb
    % f = frequency
    % alpha= LW constant, --> [0,2]
    % niter= number of iterations --> any non negative number 
    % LW_iter = number of Landweber iterations --> any non negative number 
    % GS = State vector Green's function --> NxN
        % (N= number of unknown pixel)
        % GS can be dense or sparse
    % GD = Data vectore Green's function --> NXM       
        % (M= number of measurement data) 
    % masking= masking tensor --> Nx1 column vector matrix 
        % =1, for the pixels which need to be masked
% fields
    % E_ref_o= reference E fields --> NxM
    % E_s = measured scattered field data --> Mx1
%% extraction of the parameters
f = params.f;
eb= params.eb;
GS= params.GS;
GD= double(params.GD);
alpha= params.alpha;
masking = params.masking;
niter= params.niter;
LW_iter= params.LW_iter;

%% extraction of the fields
E_ref_o= double(fields.E_ref_o);
E_s= fields.E_s;

%% initialization and pre processing
e0= 8.854e-12;
u0= pi*4e-7;
[N, n_src]= size(E_ref_o);
M= length(E_s);
masking= diag(1- masking);
%% reordering E_o
E_ref_o_full=[];
for antenna_idx= 1:n_src
    E_ref_o_full= [E_ref_o_full, repmat(E_ref_o(:,antenna_idx),1,n_src-antenna_idx)];
end
x= zeros(N,1);
y= E_s;
%% singular value decomposition for determining alpha, using background E field,one time operation to save memory
A= -1* GD.' .* E_ref_o_full.' *masking;
omega = alpha / norm(A)^2; %0<= omega <= 2/||A||^2
%% beginning iteration
tic
disp ('beginning iteration BIM with LandWeber regularization...')
error= 1e5;
tol= 1e-3;

% the first order born approximation
iter=0;
E_o_iter= E_ref_o_full;
ref_y=y;
while(error>tol)
    iter= iter+1;
    prev_x=x;
    
    % the forward problem
    E_o_iter= double(E_ref_o_full -  GS*diag(x)*E_o_iter); %%new
    current_y =-GD.' .* E_o_iter.' *x;

    % The inverse problem using landweber iterations
    omega= omega/1.125; % dampen to speed up
    A= -1* GD.' .* E_o_iter.' *masking;
    x = LW(prev_x,A,ref_y,omega,LW_iter);
    current_x=x;


    % defining MSE
    MSE(iter) = sqrt((prev_x-current_x)'*(prev_x-current_x)/(current_x'*current_x));
    error= MSE(iter);

    % defining RRE
    RRE(iter)= sum(abs(current_y- ref_y))/sum(abs(ref_y));

    disp(['iteration ' num2str(iter) ' completed. RRE= ' num2str(RRE(iter)) ' MSEconv= ' num2str(MSE(iter)) ]);
    if iter>niter
        disp ('Iteration could not converge, considering lowering alpha and beta')
        break
    end
end

fields_out.E_o_new= E_o_iter;
fields_out.E_s_new= current_y;
params_out.MSE= MSE;
params_out.RRE= RRE;
contrast = x;
disp(['BIM iterations completed, time taken= ' num2str(toc) ' s'])
end

function x_new = LW(x_old, A, b, omega, maxIter)
    x_k = x_old;
    AtA= A'*A;
    Atb= A'*b;
    for k = 1:maxIter             
        x_k = x_k + omega*(Atb - AtA*x_k);  
    end
    x_new = x_k;
end
