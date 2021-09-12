% This script approximates some matricized tensor data as a nnSuKro
% The approximation quality is compared to that of a nnCPD directly on the
% tensor data.
% factors D*X, where D is constrained to be a SuKro.
%
% Results: the approximation error (||D_out-D_in||_F/||D_in||_F ) is plotted
% as a function of the approximation ranks (figure 1) and as a function of
% the total number of parameters on its new representation (figure 2)


addpath ../misc/ ../
% Include tensorlab toolbox (insert your local path here)
tensorlab_path = '~/source/Backup/PhD/SuKro/ho-sukro-icassp2019/src/tensorlab_2016-03-28/';
assert(isfolder(tensorlab_path),'Please insert a valid local path for tensorlab toolbox')
addpath(tensorlab_path) 

rng(1)

%% Creating data
R_vec = [1 3 5 10 11]; % nb kronecker summing terms. Inf leads to unconstrained case
exp_type = 'Moffett'; % Options: 'Moffett', 'Madonna' or 'Synthetic'
switch exp_type   
    case 'Moffett'        
        load('Aviris_Moffet.mat')
        Y = double(reshape(im(:,:,:),size(im,1)*size(im,2),size(im,3)));        
        if min(Y(:)) < 0,  Y = Y - min(Y(:)); end %avoid negative entries. Or: Y(Y<0) = 0;
        % Suppression de bandes de fréquences sans énergie (pre-processing utilisé par Nicolas)
        mask = [1:4 104:115 151:175 205:222]; 
        Y(:,mask) = [];

        I = 2; % nb modes
        n = [size(im,1) size(im,2)]; %spatial dimensions
        m = [15 11]; %factorized spectral dimension (165) %[165 1]; is worse
        %I = 3;
        %n = [size(im,1) size(im,2) size(Y,2)]; %spatial dimensions
        %Y = Y(:);
        %m = [1 1 1]; %factorized spectral dimension (165)
        clear im
        
    case 'Madonna'
        load Hyspex_Madonna
        Y = double(reshape(im(:,:,:),size(im,1)*size(im,2),size(im,3)));
        
        
        %n = [size(im,1) size(im,2)];
        %m = [16 10]; %factorized spectral dimension (160)
        n = [size(im,1) size(im,2) size(Y,2)];
        m = [1 1 1]; %factorized spectral dimension (160)        
        clear im  
        
    otherwise
        I = 3; % nb modes
        n = [3 3 3]; % sizes of factors D{i,p} is nixmi for any p
        m = [2 2 2];

        % Random data Y
        Y = abs(randn([prod(n) prod(m)])); % as matrix        
end

Y_tensor = reshape(Y,[n prod(m)]);

%% Approx
diff_CPD = zeros(1,length(R_vec));
diff_SuKro = zeros(1,length(R_vec));
legend_str = cell(1,2*length(R_vec));

for Rk = 1:length(R_vec) % Run with different ranks
    R = R_vec(Rk);
    fprintf('\n R = %d',R);    
    %rng(1)

    % SuKro
    [D_ip, trace] = nnSuKroApprox(Y,n,m,R);
    diff_SuKro(Rk) = trace.diff;

    % nnCPD
    % initialize the model
    model = struct; 
    % add the data to the model "nonnegative"
    model.factorizations.nonnegative.data = Y_tensor;
    %variables
    dim = size(Y_tensor);
    model.variables.u   = rand(size(Y_tensor,1),R);  % variable u 
    model.variables.l   = rand(size(Y_tensor,2),R);  % variable l
    model.variables.a   = rand(size(Y_tensor,3),R);  % variable a
    %factors
    model.factors.U   = {'u', @struct_nonneg};  % declare nonnegative factor U dependend on variable u
    model.factors.L   = {'l', @struct_nonneg};  % declare nonnegative factor L dependend on variable l
    model.factors.A   = {'a', @struct_nonneg};  % declare nonnegative factor A dependend on variable a
    % add a CPD to the model.
    model.factorizations.nonnegative.cpd  = {'U','L','A'};

    % solve the model
    sol = sdf_nls(model);
    % extract the factors
    f = sol.factors;
    Uhat = {f.U, f.L, f.A};
    Y_CPD = cpdgen(Uhat);
    diff_CPD(Rk) = norm(Y_CPD(:) - Y_tensor(:),'fro');

end
%% Plots
normY = norm(Y,'fro');
% Approx error vs. rank
figure(1), hold on, xlabel('Rank'), ylabel('Relative Approximation Error')
semilogy(R_vec,diff_SuKro/normY)
semilogy(R_vec,diff_CPD/normY), legend('SuKro','nnCPD')

% Approx error vs. nb parameters
nparams_SuKro = sum(n.*m)*R_vec;
nparams_CPD = sum(size(Y_tensor))*R_vec;
figure(2), hold on, xlabel('Nb. of parameters'), ylabel('Relative Approximation Error')
semilogy(nparams_SuKro,diff_SuKro/normY)
semilogy(nparams_CPD,diff_CPD/normY), legend('SuKro','nnCPD')