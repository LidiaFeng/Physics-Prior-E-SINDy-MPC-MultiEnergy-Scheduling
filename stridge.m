function Xi = stridge(A, D, lambda, ridge)
% Sequential Threshold Ridge regression (STRidge), shared solver for SINDy family.
% A: regression matrix, D: targets (each state in one column), lambda: sparsity threshold, ridge: ridge coefficient.
% Both regression matrix and targets are RMS-normalized first (threshold has consistent meaning in dimensionless space).
% Restore scaling before return; coefficients remain in the original input coordinate system.

if nargin < 4, ridge = 1e-6; end

colScale = sqrt(mean(A.^2, 1));
colScale(colScale < 1e-12) = 1;
An = A ./ colScale;
dScale = sqrt(mean(D.^2, 1));
dScale(dScale < 1e-12) = 1;
Dn = D ./ dScale;

p = size(An,2);
q = size(Dn,2);
Xi = (An'*An + ridge*eye(p)) \ (An'*Dn);

for it = 1:10
    small = abs(Xi) < lambda;
    XiOld = Xi;
    Xi(small) = 0;
    for j = 1:q
        keep = ~small(:,j);
        if ~any(keep)
            % Fallback: retain the most correlated term if all candidates are pruned.
            [~, ix] = max(abs(An'*Dn(:,j)));
            keep(ix) = true;
        end
        Ak = An(:,keep);
        Xi(keep,j) = (Ak'*Ak + ridge*eye(nnz(keep))) \ (Ak'*Dn(:,j));
    end
    if isequal(abs(Xi)>0, abs(XiOld)>0), break; end
end

Xi = (Xi ./ colScale') .* dScale;
end
