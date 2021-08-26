function D_ip = nnSuKroUpdateCPD(X,Y,n,m,R,varargin)
% >>> Iterative Projection approach for optimizing NON-NEGATIVE HO-SuKro terms. <<<
% This function updates the factor D within a model Y=DX, given Y and X
% under a SuKro (sum of Kroneckers) constraint :
%     D = \sum_{p=1:R} D{1,p} ⊗ D{2,p} ⊗ ... ⊗ D{I,p}
%
% It updates the D without SuKro constraints, followed by a projection onto
% the SuKro model.
% The projection step consists in taking the non)negative CPD of a
% rearranged version of D.
%
% Inputs :
% - X : Right factor as a matrix (m1m2m3, N) or as a tensor (m1,m2,m3,N)
% - Y : Input dataset as a matrix (n1n2n3, N) or as a tensor (n1,n2,n3,N)
% - Sizes of factors D{i,p} is nixmi for any p (stocked in memory)
%       n = [n1 n2 n3]; 
%       m = [m1 m2 m3]; % where I = length(n) = length(m)
% - R : Number of Kronecker summing terms
% Optional inputs:
% - D_ip : (R, I) cell array containing SuKro terms D{1,p} (initialization)
% 
% Parameters :
% - I : Number of modes, with I = length(n) = length(m)
% - beta : beta-divergence parameter for data-fidelity (default beta = 1)
% - Stopping criteria:
%       - N_iter : maximum number of iterations (default = 30)
%       - tol : tolerance on the frobenius norm variation on each block D_ip (default = sqrt(m.*n)*1e-4)
% - verbose : activate verbose trace mode.

% addpath ../tensorlab_2016-03-28/

%% Parameters
verbose = true;

% dimensions
assert(length(n)==length(m),'Vectors n and m must have the same length.')
I = length(n);

N = size(X,ndims(X));
assert( size(Y,ndims(Y)) == N, 'Last dimension of X and Y must match in size.' )

% X and Y
assert( (ismatrix(X) && size(X,1)==prod(m)) || (ndims(X)==I+1 && all(size(X)==[m N])), ...
        'Factor X must be either a (m1m2m3, N) or a (m1,m2,m3,N) tensor.' )
if ndims(X)==I+1, X = unfold(X,ndims(X)).'; end

assert( (ismatrix(Y) && size(Y,1)==prod(n)) || (ndims(Y)==I+1 && all(size(Y)==[n N])), ...
        'Factor Y must be either a (n1n2n3, N) or a (n1,n2,n3,N) tensor.' )
if ndims(Y)==I+1, Y = unfold(Y,ndims(Y)).'; end

% beta-divergence
beta = 1; 

if beta < 1
   gamma = 1/(2-beta); 
elseif beta <= 2
    gamma = 1;
else
    gamma = 1/(1-beta);
end

% Convergence measures
N_iter = 30; % maximum number of iterations
% N_inner = 30; % number of inner iterations (for each block)
% obj = zeros(1,N_iter);

tol = 1e-4*sqrt(prod(m)*prod(n));
diff = zeros(I,R); % Frobenius norm of update on each D_ip
converged = false;

%Nonnegative CPD model initialization
% initialize the model
model = struct; 
% add the data to the model "nonnegative"
model.factorizations.nonnegative.data = abs(randn(n.*m)); %R_D;
%variables
model.variables.u   = rand(n(1)*m(1),R);  % variable u 
model.variables.l   = rand(n(2)*m(2),R);  % variable l
model.variables.a   = rand(n(3)*m(3),R);  % variable a
%factors
model.factors.U   = {'u', @struct_nonneg};  % declare nonnegative factor U dependend on variable u
model.factors.L   = {'l', @struct_nonneg};  % declare nonnegative factor L dependend on variable l
model.factors.A   = {'a', @struct_nonneg};  % declare nonnegative factor A dependend on variable a
% add a CPD to the model.
model.factorizations.nonnegative.cpd  = {'U','L','A'};

%D initialization
D_ip =  cell(I,R);
if nargin > 5 % provided as an input
    if iscell(varargin{1}) && all(size(varargin{1})==[I,R])
        D_ip = varargin{1};
        D = zeros(size(prod(n),prod(m)));
        for p = 1:R
%             D = D + kron(D_ip(1:I,p));
            D = D + kron(D_ip(I:-1:1,p));
        end        
    elseif ismatrix(varargin{1}) && all(size(varargin{1})==[prod(n),prod(m)])
        D = varargin{1};
    else
        error('Initialization for D should be provided either as a (prod(n), prod(m)) matrix or a (I,R) cell array')
    end
else % random initialization
    D = abs(randn(prod(n),prod(m)));
end


%% Optimizing dictionary, given X and Y

k_ALS = 0;
while ~converged && k_ALS <= N_iter, k_ALS = k_ALS + 1;
    
    if verbose, fprintf('%4d,',k_ALS); end
  
    D_old = D;
    
    % Conventional update on D (no SuKro constraint)
    % --- Multiplicative update (beta divergences) ---
    D = D.*( (((D*X).^(beta-2).*Y)*X.') ./ ((D*X).^(beta-1)*X.') ).^gamma;
    % --- Non-negative LS (beta=2 only) ---
    % TODO! Not implemented yet. Call a pre-existing implementation.

    % Projection into SuKro model (sum of alpha separable terms)
    R_D = rearrangement_recursive(D,n,m); % Rearrangement
    
    %Standard CPD
    % Recalculating CPD from scratch
%         Uhat = cpd(R_D,R); % supposing underlying rank (alpha) known
    % Providing initialization to CPD
%     if ~exist('Uhat','var') % first iteration
%         Uhat = cpd(R_D,R); % supposing underlying rank (alpha) known
%     else 
%         Uhat = cpd(R_D,Uhat); % initializing with previous estimation
%     end

    %Nonnegative CPD
    % add the data to the model "nonnegative"
    model.factorizations.nonnegative.data = R_D;
    % solve the model
    sol = sdf_nls(model);
    % extract the factors
    f = sol.factors;
    Uhat = {f.U, f.L, f.A};
        
    D = rearrangement_inv_recursive(cpdgen(Uhat),n,m);

    % compute the relative error
%     ['The relative error of the approximation is ' num2str(frob(R_D - cpdgen(Uhat)) / frob(R_D))]
    diff = norm(D_old-D,'fro');


    % Stop Criterion
    if verbose, diff, end
    if ( (diff < tol) || (k_ALS >= N_iter) )
        converged = true;
        % disp(['Total nº of iterations: ' num2str(iter)]);
    end

    % Calculate the objective function
    % Euclidean
%     obj(k_ALS) = norm(Y - D*X,'fro');
    % Beta divergence
    % not implemented!
end

for i0 = 1:I
    for p0 = 1:R
        D_ip{i0,p0} = reshape(Uhat{i0}(:,p0),[n(i0) m(i0)]);
    end
end
