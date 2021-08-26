% Implements a generalization for a kronecker product of higher orders
% (more than 2 matrices) of rearrangement defined in [1]. It is, more
% precisely, the inverse rearrangement corresponding to a transpose version
% of the generalized rearrangement.
% The first dimension of the rearranged version corresponds to the
% vectorization of the last term on the kronecker product.
% So, for an input matrix D = kron(A,B,C,...Z) being the kronecker product
% of K matrices (A,B,...Z) with sizes (nA,mA), (nB,mB), ... (nZ,mZ), then D 
% has size (nA*nB*...nZ, mA*mB*...*mZ) and the rearranged version R_D is  
% order-K tensor with size (nZ*mZ,...,n2*m2, n1*m1).
%
%  [1] C.F. Dantas, M. N. da Costa, R.R. Lopes, "Learning Dictionaries as a
%      sum of Kronecker products", Signal Processing Letters 2017.
function D = rearrangement_inv_recursive(R_D,n,m)
assert(length(n)==length(m))

if length(n)==1 % Base case
    D =  reshape(R_D,n(1),m(1)); % Unvectorizes a given block.
else
    D = [];
    
    % Go over each block of the matrix.
    % Then recursively go over all sub-blocks in the block.
    for j1 = 1:m(1)
        D_col = [];
        for i1 = 1:n(1)
            % Indexes that correspond to the block to be reconstructured
            numel_block = prod(n(2:end))*prod(m(2:end));
            in_idx = (1:numel_block) + ((i1-1) + (j1-1)*n(1))*numel_block;
            % Unvectorizes the block and concatenates with previous ones
            D_col = cat(1,D_col,rearrangement_inv_recursive(R_D(in_idx),n(2:end),m(2:end)));
            %D = [D; rearrangement_inv_recursive];
        end
        D = cat(2,D,D_col);
        %D = [D D_col];
    end
end
