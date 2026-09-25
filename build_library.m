function [Theta, names] = build_library(X, U, usePrior)
% Candidate function dictionary, shared by the three methods.
% Variables are first normalized by fixed physical ranges, ensuring coefficients identified from different training sets lie within the same coordinate system.
% Subsequent coefficient drift and parameter recovery are directly compared under this coordinate frame.
%
% Dictionary composition: constant term + 9 linear terms + 45 quadratic terms + 2 physics-informed prior terms (power balance, source-load balance).
% When usePrior=false, the two prior columns are removed. They are exact linear combinations of linear columns,
% which destroy the uniqueness of true coefficients in parameter recovery experiments (rank-deficient design matrix) and must be excluded.

if nargin < 3, usePrior = true; end
sc = lib_scales();
Z = [X, U] ./ sc;          % n by 9, unified coordinate system

[n, d] = size(Z);
nQuad = d*(d+1)/2;
nCol = 1 + d + nQuad;
if usePrior, nCol = nCol + 2; end
Theta = zeros(n, nCol);
Theta(:,1) = 1;
Theta(:, 2:1+d) = Z;

vars = {'hyd','gen','pump','net','load','wind','pv','sin','cos'};
names = cell(1, size(Theta,2));
names{1} = '1';
for i = 1:d, names{1+i} = vars{i}; end

c = 1 + d;
for i = 1:d
    for j = i:d
        c = c + 1;
        Theta(:,c) = Z(:,i) .* Z(:,j);
        names{c} = [vars{i} '*' vars{j}];
    end
end

if usePrior
    % Physics priors: residual of instantaneous power balance and residual of source-load balance (also under the unified coordinate system)
    Theta(:, c+1) = (X(:,1) + X(:,2) - X(:,3) - X(:,4)) / sc(4);
    Theta(:, c+2) = (U(:,1) - U(:,2) - U(:,3) - X(:,4)) / sc(4);
    names{c+1} = 'balance';
    names{c+2} = 'netgap';
else
    names = names(1:c);
end
end
