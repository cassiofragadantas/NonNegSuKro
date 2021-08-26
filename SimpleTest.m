%TODOs
% - output objective
% - add params (beta, N_iter, verbose, compute_obj)

addpath ./misc/
% Include tensorlab toolbox (insert your local path here)
tensorlab_path = '~/source/Backup/PhD/SuKro/ho-sukro-icassp2019/src/tensorlab_2016-03-28/';
assert(isfolder(tensorlab_path),'Please insert a valid local path for tensorlab toolbox')
addpath(tensorlab_path) 


rng(1)
verbose = true;

%% Creating data
I = 3; % nb modes
R = 3; % nb kronecker summing terms

% sizes of factors D{i,p} is nixmi for any p
% Simple experiment
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

% Initial estimation of D (random)
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

%profile on
tic
D_ip = nnSuKroUpdateBCD(X,Y,n,m,R,D_ip);
%D_ip = nnSuKroUpdateCPD(X,Y,n,m,R,D);
toc
%profile off, profsave(profile('info'),'myprofile_results')


%% Dictionary error (after kron)

D = zeros(size(prod(n),prod(m)));
for p = 1:R
%     D = D + kron(D_ip(1:I,p));
    D = D + kron(D_ip(I:-1:1,p));
end

D2 = zeros(size(prod(n),prod(m)));
for p = 1:R
%     D = D + kron(D_ip(1:I,p));
    D2 = D2 + kron(D_ip(I:-1:1,p));
end

norm(D_oracle - D, 'fro')

%% Objective function
calc_obj = false;
if calc_obj
    Y_r = zeros([n N]);
    for p=1:R
        %Y_r = Y_r + tmprod(X,D_ip(1:I,p),fliplr(1:I)); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
        Y_r = Y_r + tmprod(X,D_ip(1:I,p),1:I); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
    end

    obj = norm(Y(:)-Y_r(:),'fro')
end

%% Plotting results
subplot(2,R+1,[1 2]*(R+1)), semilogy(obj)
% show dictionaries
for i = 1:I
    for p = 1:R
        subplot(2,I*R,(p-1)*I+i), imagesc(D_ip{i,p})
        subplot(2,I*R,I*R+(p-1)*I+i), imagesc(D_ip_oracle{i,p} )
    end
end

figure, subplot(2,R+1,[1 2]*(R+1)), semilogy(obj)
% show terms (sub-dictionaries)
for p = 1:R
    subplot(2,R+1,p), imagesc(kron(D_ip(1:I,p)))
    subplot(2,R+1,p+R+1), imagesc(kron(D_ip_oracle(1:I,p)))    
end

%% Measure RC
measure_RC = false;
if measure_RC
    % Preparation for RC calculation
    D_terms = D_ip;
    cpd_rank = R;
    D = randn(prod(n),prod(m));
    RC
end

%% OMP
measure_OMP_RC = false;
if measure_OMP_RC
    sparsity = 3;
    % Calculate full dictionary matrix
    D = zeros(prod(n),prod(m));
    for p = 1:R 
        D = D + kron(D_ip(I:-1:1,p));
    end
    normVec = sqrt(sum(D.^2));
    D = D./repmat(normVec,size(D,1),1);
    % D = bsxfun(@rdivide,D,normVec); % apparently slower

    X = reshape(X,[prod(m), N]);
    Y = reshape(Y,[prod(n), N]);

    tic
    profile on
    for k = 1:N
        [X(:,k), ~] = omp_Remi(Y(:,k),D,sparsity);
    end
    profile off
    profsave(profile('info'),'myprofile_OMP_naive_tensor')
    OMP_time = toc


    tic
    profile on
    for k = 1:N
        [X(:,k), ~] = omp_tensor(Y(:,k),D,sparsity, D_ip,normVec.',n,R);
    end
    profile off
    profsave(profile('info'),'myprofile_OMP_naive')
    OMP_tensor_time = toc


    n_atoms = prod(m);
    X2 = zeros(size(X));
    tic
    profile on
    for k = 1:N
        [X2(:,k), ~] = SolveOMP(D,Y(:,k),n_atoms,sparsity); % next parameters in order: lambdaStop, solFreq, verbose, OptTol
    end
    profile off
    profsave(profile('info'),'myprofile_OMP_Cholesky')
    OMP_time = toc

    n_atoms = prod(m);
    X2 = zeros(size(X));
    tic
    profile on
    for k = 1:N
        [X2(:,k), ~] = SolveOMP_tensor(D,Y(:,k),n_atoms,D_ip,normVec.',n,R,sparsity); % next parameters in order: lambdaStop, solFreq, verbose, OptTol
    end
    profile off
    profsave(profile('info'),'myprofile_OMP_Cholesky_tensor')
    OMP_tensor_time = toc
end 