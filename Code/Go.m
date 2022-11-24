% max tr(W'AW), subject to W'W=I, ||W||_2,0 <= k
% W is d-by-m, A is d-by-d
% Require: m<=k<=d, rank(A)<=m

% function [W, opt_index] = Go(A, m, k)
function [W, opt_index] = Go(A, m, k)

[d, ~] = size(A);

if rank(A) > m
    [vec, val] = eigs(A, m);
    A = vec * val * vec';
end

[~, ind] = sort(diag(A), 'descend');
opt_index = sort(ind(1:k));

W = zeros(d, m);
Aopt = A(opt_index, opt_index);
[V, ~] = eigs(Aopt, m);

W(opt_index, :) = V;

end
