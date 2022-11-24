% max tr(W'AW), subject to W'W=I, ||W||_2,0 <= k
% W is d-by-m, A is d-by-d
% Require: m<=k<=d

function [W, opt_index] = IPU(A, dim, m, k, W0)


NITER = 10;

if nargin < 4

    [W0, ~] = Go(A, m, k);
end

W = W0;

if rank(A) <= m
    [W, opt_index] = Go(A, m, k);
%     W = W0;
    return;
end
for iter = 1:NITER
    
   P = A*W*pinv(W'*A*W)*W'*A;
   [~, ind] = sort(diag(P), 'descend');
   opt_index = sort(ind(1:k));
   Aopt = A(opt_index, opt_index);
   [V, ~] = eigs(Aopt, m);
   W = zeros(dim,m);
   W(opt_index, :) = V;
   
end
%     ob = trace(W'* A * W)

end