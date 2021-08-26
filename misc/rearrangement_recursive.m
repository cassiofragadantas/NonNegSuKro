% Implements a generalization for a kronecker product of higher orders
% (more than 2 matrices) of rearrangement defined in [1]. It is, more
% precisely, the transpose version of such generalized rearrangement.
% The first dimension of the rearranged version corresponds to the
% vectorization of the last term on the kronecker product.
% So, for an input matrix D = kron(A,B,C,...Z) being the kronecker product
% of K matrices (A,B,...Z) with sizes (nA,mA), (nB,mB), ... (nZ,mZ), then D 
% has size (nA*nB*...nZ, mA*mB*...*mZ) and the rearranged version R_D is  
% order-K tensor with size (nZ*mZ,...,n2*m2, n1*m1).
%
%  [1] C.F. Dantas, M. N. da Costa, R.R. Lopes, "Learning Dictionaries as a
%      sum of Kronecker products", Signal Processing Letters 2017.

function R_D = rearrangement_recursive(D,n,m)
assert(length(n)==length(m))

%TODO add case where the rearrangement indexes are given

if length(n)==1 % Base case
    R_D =  D(:); % Vectorizes a given block.
else
    R_D = [];
    
    % Go over each block of the matrix.
    % Then recursively go over all sub-blocks in the block.
    for j1 = 1:m(1)
        for i1 = 1:n(1)
            % Indexes of the block to be treated
            in_rows  = (1:prod(n(2:end)))+(i1-1)*prod(n(2:end));
            in_cols  = (1:prod(m(2:end)))+(j1-1)*prod(m(2:end));
            % Reorders the block and concatenates the results
            R_D = cat(length(n),R_D,rearrangement_recursive(D(in_rows,in_cols),n(2:end),m(2:end)));
        end
    end
end
