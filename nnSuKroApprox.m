function [D_ip, trace] = nnSuKroApprox(D,n,m,R,varargin)
% >>> Projection onto the space of SuKro matrices <<<
% This function approximates an input matrix D as a sum of Kronecker 
% products (SuKro):
%     D = \sum_{p=1:R} D{1,p} ⊗ D{2,p} ⊗ ... ⊗ D{I,p}
%
% The projection consists in taking the non-negative CPD of a (properly) 
% rearranged version of D.
%
% Inputs :
% - D : (n1n2n3, m1m2m3) matrix to be approximated as a SuKro
% - Sizes of factors D{i,p} is nixmi for any p (stocked in memory)
%       n = [n1 n2 n3]; 
%       m = [m1 m2 m3]; % where I = length(n) = length(m)
% - R : Number of Kronecker summing terms
% Optional inputs:
% - params : struct containing customizable parameters
%       - verbose : activate verbose trace mode (default = true)
%       - trace_on : activate computation of trace variables (default = false)
% 
% Outputs:
% - D_ip : (R, I) cell array containing optimized SuKro terms D{1,p}
% - trace : struct containing set of trace variables
%       - .diff : approximation error (euclidean)


% addpath ../tensorlab_2016-03-28/

%% Parameters

% dimensions
assert(length(n)==length(m),'Vectors n and m must have the same length.')
I = length(n);
if I == 2, n = [n 1]; m=[m 1]; end


% D
assert( (ismatrix(D) && all(size(D)==[prod(n) prod(m)])), ...
        'Input D must be a matrix of size (prod(n), prod(m)).' )


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

% Customizable params
params = struct;
if nargin > 4 
    params = varargin{2};
    assert(isstruct(params), 'Optional input params must be a struct.')
end

%verbose = true;
%if isfield(params,'verbose'), verbose = params.verbose; end

trace_on = true;
if isfield(params,'trace_on'), trace_on = params.trace_on; end 
trace = struct;



%% Projection into SuKro model (sum of alpha separable terms)
D_old = D;

R_D = rearrangement_recursive(D,n(end:-1:1),m(end:-1:1)); % Rearrangement

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
R_D_tilde = cpdgen(Uhat); %norm(R_D_tilde(:)-R_D(:))

D = rearrangement_inv_recursive(cpdgen(Uhat),n(end:-1:1),m(end:-1:1));

D_ip = cell(I,R);
for i0 = 1:I
    for p0 = 1:R
        D_ip{i0,p0} = reshape(Uhat{i0}(:,p0),[n(i0) m(i0)]);
    end
end

% Calculate approximation error
if trace_on
    % Euclidean
    trace.diff = norm(D_old-D,'fro');
    % Beta divergence
    % not implemented!
end
