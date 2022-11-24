function [W, opt_index, b, obj] = UFPCA(X, m, k, p, h, NITER)

% X is data matrix, each column is a data
% m reduced dimension 
% k selected dimension
% p norm number
% h parameter, the number of normal data
% NITER the number of the iteration

% X = X';
[dim, n] = size(X) 

if nargin <= 2
    NITER = 10;
end;
a =[zeros(1,n-h),ones(1,h)];       
s = a(randperm(size(a,2))); 

W = orth(rand(dim, m));
b = mean(X,2);
e = ones(1,n);

obj = zeros(NITER,1);
ob = zeros(NITER,1);

for iter = 1:NITER
    S = spdiags(sqrt(s'),0,n,n);     %对S进行对角化

    % calculate data without mean   
    A = X - b*e;          
    Do = A - W*W'*A;
   
    d = 0.5 * p * (sqrt(sum(Do.*Do,1)+eps)).^(p-2);    %d是一个行向量
    D = spdiags(d',0,n,n);     %对D进行对角化
    
    Ds = D * S;
    % calculate a positive-semidefinite matrix
    Y = X * (Ds - (sum(Ds, 2) * sum(Ds, 1))/(d * s')) * X';
    % calculate the value of subproblem
    ob(iter) = trace(W'* Y * W);
    % updata W
    [W, opt_index]= IPU(Y, dim, m, k, W);
    % update b (mean)
    b = (X * sum(Ds, 2))/(d * s');
    % calculate the value of objective function
    Bi = (sqrt(sum(Do.*Do,1)+eps).^p);
    obj(iter) = Bi * s';
    % update S
    % Bo = sum(Do.*Do,1)+eps;
    [~,index] = sort(Bi,'ascend');       
    in = index(1:h);
    s = zeros(1,n);
    s(in)=1;
end


