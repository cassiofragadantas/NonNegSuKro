% This script generates random factors X and D_oracle and computes Y = D_oracle*X. 
% Then, factor D is optimized given X and Y from a random initialization
% (using function nnSuKroUpdateBCD or nnSuKroUpdateCPD).
% The obtained factor D is compared to D_oracle, and the reconstruction D*X
% is compared to D_oracle*X.

addpath ./misc/
% Include tensorlab toolbox (insert your local path here)
tensorlab_path = '~/source/Backup/PhD/SuKro/ho-sukro-icassp2019/src/tensorlab_2016-03-28/';
assert(isfolder(tensorlab_path),'Please insert a valid local path for tensorlab toolbox')
addpath(tensorlab_path) 

%rng(1)

%% Creating data
I = 3; % nb modes
R = 3; % nb kronecker summing terms

% sizes of factors D{i,p} is nixmi for any p
n = [2 2 2]; % size I
m = [3 3 3];

N = 15; % Number of training samples

% Array containing all factors D_ip
D_ip_oracle = cell(I,R);

% Randomly initialize D_ip and X
for i = 1:I
    for p = 1:R
        D_ip_oracle{i,p} = abs(randn(n(i),m(i)));
    end
end

%unfoldin the Kronecker products
D_oracle = zeros(size(prod(n),prod(m)));
for p = 1:R
%     D_oracle = D_oracle + kron(D_ip_oracle(1:I,p));
    D_oracle = D_oracle + kron(D_ip_oracle(I:-1:1,p));
end


% Initialize X (as tensor)
X = abs(randn([m N])); % random dense
% density = 10/N;
%X = sprand(prod(m),N,density); % random sparse
% X = reshape(full(X),[m N]); % sparse tensor not supported

% Initialize Y (as tensor)
Y = zeros([n N]);
for p=1:R
%     Y = Y + tmprod(X,D_ip_oracle(1:I,p),fliplr(1:I)); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
    Y = Y + tmprod(X,D_ip_oracle(1:I,p),1:I); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
end
%Y = D_oracle*unfold(X,4).'; % as matrix, same as unfold(Y,4).'
%Y = randn([n N]); % Random

%% Optimizing dictionary, given X and Y

% Random initialization for D (random)
D_ip =  cell(I,R);
for i = 1:I
    for p = 1:R
        D_ip{i,p} = abs(randn(n(i),m(i)));
    end
end

% Easier test : only one D_ip wrong and test one iteration
% i0 = 1;
% p0 = 1;
% D_ip =  cell(I,R);
% for i = 1:I
%     for p = 1:R
%         if (i==i0) && (p==p0)
%             D_ip{i,p} = randn(n(i0),m(i0));
%         else
%             D_ip{i,p} = D_ip_oracle{i,p};
%         end
%         
%     end
% end

% ========= SuKro optimization ==========
% parameters (optional)
params = struct;
params.trace_on = true;
params.N_iter = 20000;
params.rel_tol = 1e-5;
%params.verbose = false;
%params.beta = 1;

tic, [D_ip, trace] = nnSuKroUpdateBCD(X,Y,n,m,R,D_ip,params); toc
% tic, [D_ip, trace] = nnSuKroUpdateCPD(X,Y,n,m,R,D,params); toc


%% Reconstruction errors
% Dictionary reconstruction error

D = zeros(size(prod(n),prod(m)));
for p = 1:R
%     D = D + kron(D_ip(1:I,p));
    D = D + kron(D_ip(I:-1:1,p));
end

fprintf('Relative reconstruction error on D: %f \n', norm(D_oracle - D, 'fro')/norm(D_oracle,'fro'))

% Input signal
Y_r = zeros([n N]);
for p=1:R
    %Y_r = Y_r + tmprod(X,D_ip(1:I,p),fliplr(1:I)); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
    Y_r = Y_r + tmprod(X,D_ip(1:I,p),1:I); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
end

fprintf('Relative reconstruction error on Y(=DX): %f \n',norm(Y(:)-Y_r(:),'fro')/norm(Y(:),'fro'))

%% Plotting results

figure
% show dictionaries
for i = 1:I
    for p = 1:R
        subplot(2,I*R,(p-1)*I+i), imagesc(D_ip_oracle{i,p})
        subplot(2,I*R,I*R+(p-1)*I+i), imagesc(D_ip{i,p} )
    end
end

figure
% show terms (sub-dictionaries)
for p = 1:R
    subplot(2,R,p), imagesc(kron(D_ip_oracle(1:I,p)))
    subplot(2,R,p+R), imagesc(kron(D_ip(1:I,p)))
end

%Objective function
if exist('params','var') && isfield(params,'trace_on') && params.trace_on
    obj = trace.obj;
else
    obj = 1;
end
figure, semilogy(obj)
xlabel('Iteration'), ylabel('Squared error')

%% Measure RC
measure_RC = false;
if measure_RC
    % Preparation for RC calculation
    D_terms = D_ip;
    cpd_rank = R;
    D = randn(prod(n),prod(m));
    RC
end