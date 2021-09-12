% This script generates random data Y and factorizes it into two non-negative
% factors D*X, where D is constrained to be a SuKro.
%
% Results: the relative error (||Y-DX||_F/||Y||_F ) evolution over the 
% iterations is plotted for different SuKro ranks (nb. of summing Kronecker 
% terms) and is compared to the unconstrained case (D not SuKro).
%
% Comments:
% - Potential problem for sparse NMF (L1 regularization for factor X), some
%   kind of column normalization is necessary for D. This can be
%   problematic for SuKro (normalized blocks don't guarantee normalized
%   sum). However, probably normalized blocks suffice, since it upper
%   bounds the column norms.

addpath ../misc/ ../
% Include tensorlab toolbox (insert your local path here)
tensorlab_path = '~/source/Backup/PhD/SuKro/ho-sukro-icassp2019/src/tensorlab_2016-03-28/';
assert(isfolder(tensorlab_path),'Please insert a valid local path for tensorlab toolbox')
addpath(tensorlab_path) 

rng(1)

%% Creating data
R_vec = [1 3 10 20 Inf]; % nb kronecker summing terms. Inf leads to unconstrained case
N_it = 1000; % Number of NMF iterations
beta = 2;
exp_type = 'Synthetic'; % Options: 'Moffett', 'Madonna' or 'Synthetic'
switch exp_type   
    case 'Moffett'        
        load('Aviris_Moffet.mat')
        Y = double(reshape(im(:,:,:),size(im,1)*size(im,2),size(im,3)));        
        if min(Y(:)) < 0,  Y = Y - min(Y(:)); end %avoid negative entries. Or: Y(Y<0) = 0;
        % Suppression de bandes de fréquences sans énergie (pre-processing utilisé par Nicolas)
        mask = [1:4 104:115 151:175 205:222]; 
        Y(:,mask) = [];

        I = 2; % nb modes
        n = [size(im,1) size(im,2)];
        m = [5 5];
        N = size(Y,2);      
        clear im
        
    case 'Madonna'
        load Hyspex_Madonna
        Y = double(reshape(im(:,:,:),size(im,1)*size(im,2),size(im,3)));
        
        I = 2; % nb modes
        n = [size(im,1) size(im,2)];
        m = [5 5];
        N = size(Y,2);                
        clear im  
        
    otherwise
        I = 3; % nb modes
        n = [3 3 3]; % sizes of factors D{i,p} is nixmi for any p
        m = [2 2 2];
        N = 15; % Number of training samples

        % Random data Y
        %Y = abs(randn([n N])); % as tensor
        Y = abs(randn([prod(n) N])); % as matrix        
end

%% NMF (compute D and X given Y, with Y = DX)

% D update parameters
params = struct;
params.trace_on = false;
params.N_iter = 1;
%params.rel_tol = 1e-5;
params.verbose = false;
params.beta = beta;

rel_err = zeros(1,N_it);
figure(1), hold on, xlabel('Iteration'), ylabel('Relative Approximation Error')
legend_str = cell(1,length(R_vec));

for Rk = 1:length(R_vec) % Run with different SuKro ranks
R = R_vec(Rk);
nbytes = 0; fprintf('\n R = %d : ',R);
legend_str{Rk} = ['R = ' num2str(R)];
rng(1)

% Random initialization for X
%X = abs(randn([m N])); % as tensor
X = abs(randn([prod(m) N])); % as matrix

% Random initialization for D
if R == Inf % Unconstrained
    D = abs(randn(prod(n),prod(m)));
else
    D_ip =  cell(I,R);
    for i = 1:I
        for p = 1:R
            D_ip{i,p} = abs(randn(n(i),m(i)));
        end
    end
    D = zeros(size(prod(n),prod(m)));    
    for p=1:R
        %D = D + kron(D_ip(1:I,p));
        D = D + kron(D_ip(I:-1:1,p));        
    end     
end

%% NMF
for k=1:N_it
    %Print iteration number
    fprintf(repmat('\b',1,nbytes))
    nbytes = fprintf('Iteration %d of %d\n', k, N_it);
    
    % D update 
    if R == Inf % Unconstrained
        % Multiplicative update
        D = D.*( (((D*X).^(beta-2).*Y)*X.') ./ ((D*X).^(beta-1)*X.') ); %.^gamma;        
    else % SuKro constraint
        [D_ip, trace] = nnSuKroUpdateBCD(X,Y,n,m,R,D_ip,params);
        % [D_ip, trace] = nnSuKroUpdateCPD(X,Y,n,m,R,D,params);
        
        D = zeros(size(prod(n),prod(m)));    
        for p=1:R
            %D = D + kron(D_ip(1:I,p));
            D = D + kron(D_ip(I:-1:1,p));        
        end          
    end
    
    % X update   
    Y_r = D*X;
    rel_err(k) = norm(Y(:)-Y_r(:),'fro')/norm(Y(:),'fro');
    %fprintf('Relative reconstruction error on Y(=DX): %f \n',rel_err)
    
    % Multiplicative update
    X = X.*( (D.'*(Y.*Y_r.^(beta-2))) ./ (D.'*(Y_r.^(beta-1))) ); %.^gamma     
end
%% Plots
figure(1), semilogy(rel_err)

% show dictionaries column as an image
if strcmp(exp_type,'Moffett') || strcmp(exp_type,'Madonna')
figure(2)
subplot(1,length(R_vec),Rk), imagesc(reshape(D(:,2),n)), title(['R =' num2str(R)])

figure(2+Rk), title(['R =' num2str(R)])
if R < inf
    for p =1:R
        D_term = kron(D_ip(I:-1:1,p));
        subplot(1,R,p), imagesc(reshape(D_term(:,2),n)), 
    end
end

end
end
figure(1), legend(legend_str)
