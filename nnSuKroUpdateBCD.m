function [D_ip, trace] = nnSuKroUpdateBCD(X,Y,n,m,R,varargin)
% >>> Block Coordinate Descent approach for optimizing NON-NEGATIVE HO-SuKro terms. <<<
% This function updates the factor D within a model Y=DX, given Y and X
% under a SuKro (sum of Kroneckers) constraint :
%     D = \sum_{p=1:R} D{1,p} ⊗ D{2,p} ⊗ ... ⊗ D{I,p}
%
% It updates the blocks D{i,r} alternatively (i=1,...,I), but updates
% all rank terms (p=1:R) simultaneously for a given i.
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
% - params : struct containing customizable parameters
%       - beta : beta-divergence parameter for data-fidelity (default beta = 1)
%       - Stopping criteria:
%           - N_iter : maximum number of iterations (default = 30)
%           - rel_tol : tolerance on the frobenius norm variation on each 
%                       block D_ip, tol = sqrt(m.*n)*rel_tol (default: 1e-4)
%       - update : specifies algorithm used on the block updates (default = 'MM')
%       - N_inner : number of inner iterations the MM block updates (default = 10)
%       - verbose : activate verbose trace mode (default = true)
%       - trace_on : activate computation of trace variables (default = false)
% 
% Outputs:
% - D_ip : (R, I) cell array containing optimized SuKro terms D{1,p}
% - trace : struct containing set of trace variables
%       - .obj : objective function over the iterations
%       - .time_it : execution time over the iterations


% addpath ../tensorlab_2016-03-28/

%% Parameters

% dimensions
assert(length(n)==length(m),'Vectors n and m must have the same length.')
I = length(n);

N = size(X,ndims(X));
assert( size(Y,ndims(Y)) == N, 'Last dimension of X and Y must match in size.' )

% X and Y
assert( (ismatrix(X) && size(X,1)==prod(m)) || (ndims(X)==I+1 && all(size(X)==[m N])), ...
        'Factor X must be either a (m1m2m3, N) or a (m1,m2,m3,N) tensor.' )
if ismatrix(X), X = reshape(X,[m N]); end

assert( (ismatrix(Y) && size(Y,1)==prod(n)) || (ndims(Y)==I+1 && all(size(Y)==[n N])), ...
        'Factor Y must be either a (n1n2n3, N) or a (n1,n2,n3,N) tensor.' )
if ismatrix(Y), Y = reshape(Y,[n N]); end

%D initialization
if nargin > 5 % provided as an input
    D_ip = varargin{1};
    assert(iscell(D_ip) && all(size(D_ip)==[I,R]), ...
        'Initialization for D should be provided as a (I,R) cell array')
else % random initialization
    D_ip =  cell(I,R);
    for i = 1:I
        for p = 1:R
            D_ip{i,p} = abs(randn(n(i),m(i)));
        end
    end    
end

params = struct;
if nargin > 6 % customizable params
    params = varargin{2};
    assert(isstruct(params), 'Optional input params must be a struct.')
end

verbose = true;
if isfield(params,'verbose'), verbose = params.verbose; end

% beta-divergence
beta = 2;
if isfield(params,'beta'), beta = params.beta; end

if beta < 1
   gamma = 1/(2-beta); 
elseif beta <= 2
    gamma = 1;
else
    gamma = 1/(1-beta);
end

% Convergence measures
N_iter = 30; % maximum number of iterations
if isfield(params,'N_iter'), N_iter = params.N_iter; end
N_inner = 10; % number of inner iterations (for each block)
if isfield(params,'N_inner'), N_inner = params.N_inner; end

rel_tol = 1e-4;
if isfield(params,'rel_tol'), rel_tol = params.rel_tol; end
tol = rel_tol*sqrt(m.*n);
diff = zeros(I,R); % Frobenius norm of update on each D_ip
converged = false;

% Block update type
if ~isfield(params,'update'), params.update = 'MM'; end

% Trace variables
trace_on = false;
if isfield(params,'trace_on'), trace_on = params.trace_on; end 
trace = struct;
if trace_on, trace.obj = zeros(1,N_iter); trace.time_it = zeros(1,N_iter); end

%% Optimizing D, given X and Y

k_ALS = 0;
tStart = tic;
while ~converged && k_ALS <= N_iter, k_ALS = k_ALS + 1;
    
    if verbose, fprintf('%4d,',k_ALS); end
    
    % Go through the blocks i=1...I
    for i0 = circshift(1:I,[0,-1])%[2 3 1]

        Ui0 = cell(R,1); % each Ui0 is (m(i0) x N*prod(m(1:i0-1 i0+1:I)))

        % Grouping all indexes p=1:R in a single block
        for p = 1:R 
            Ui0{p} = unfold(tmprod(X,D_ip([1:i0-1 i0+1:I],p),[1:i0-1 i0+1:I]),i0); % same as: unfold(X,i0)*kron({eye(N) D_ip{fliplr([1:i0-1 i0+1:I]),p}}).';
        end

        W = cell2mat(Ui0).';
        H = cell2mat(D_ip(i0,:)).';

        V = unfold(Y,i0).';
        
        % Block update (H given V and W)

        % ---- MU ----
        if strcmp(params.update, 'MM')
            for k = 1:N_inner
            V_tilde = W*H;
            H = H.*( (W.'*(V.*V_tilde.^(beta-2))) ./ (W.'*(V_tilde.^(beta-1))) ).^gamma;
            end
        
        % --- NN-LS --- (beta=2 only)
        else
            assert(beta==2,'NNLS block update only applies for beta=2.')
            H = nnlsHALSupdt(V,W,H,500);
        end
        
%         end

        % Compute variation
        for p0 = 1:R
            diff(i0,p0) = norm(D_ip{i0,p0}-H(m(i0)*(p0-1) + (1:m(i0)),:).','fro');
            D_ip{i0,p0} = H(m(i0)*(p0-1) + (1:m(i0)),:).';
        end
        
    end

    % Stop Criterion
    if verbose, diff, end
    if ( all(mean(diff,2) < tol.') || (k_ALS >= N_iter) )
        converged = true;
        % disp(['Total nº of iterations: ' num2str(iter)]);
    end

    % Calculate the objective function
    if trace_on
        trace.time_it(k_ALS) = toc(tStart);
        
        % reconstruction
        Y_r = zeros([n N]);
        for p=1:R
            %Y_r = Y_r + tmprod(X,D_ip(1:I,p),fliplr(1:I)); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
            Y_r = Y_r + tmprod(X,D_ip(1:I,p),1:I); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
        end
        
        % Euclidean
        trace.obj(k_ALS) = norm(Y(:)-Y_r(:),'fro'); 
        % Beta divergence
        % not implemented!
    end
    
end