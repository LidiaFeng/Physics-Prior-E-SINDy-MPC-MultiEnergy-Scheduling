function [sol, info] = admm_coord(nFc, dtv, price, PhPrev, E0, loadFc, ~, Nc, lim, u0, rho)
% Distributed coordination via ADMM. The centralized problem is decomposed into two subproblems of hydropower and pumped storage according to physical entities.
% The coupling constraint is the sequential power balance:
% Ph_k + (Pg_k - Pp_k + sp_k - sm_k) = nFc_k.
% Iteration is performed via scaled ADMM, and primal/dual residuals are recorded at each iteration.

if nargin < 11, rho = 0.05; end
H = numel(nFc);
Nc = min(Nc, H);
[T, D] = horizon_maps(H, Nc);
wdu = 1e-4; wE = 2e-4;
maxIter = 30;
tolPri = max(2*sqrt(H), 1e-3*norm(nFc));
tolDua = tolPri;

u = u0(:);
if numel(u) < H, u = [u; repmat(u(end), H - numel(u), 1)]; end
u = u(1:H);

opts = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');

% The part of the hydropower subproblem that remains unchanged across iterations.
Hh = 2*wdu*(D'*D) + rho*(T'*T) + 1e-9*eye(Nc);
Hh(1,1) = Hh(1,1) + 2*wdu;
fh0 = 11 * (T'*dtv);
fh0(1) = fh0(1) - 2*wdu*PhPrev;
% Ramp constraint
Ah = zeros(2*Nc, Nc); bh = zeros(2*Nc, 1);
Ah(1,1) = 1;  bh(1) = PhPrev + lim.ramp*dtv(1);
Ah(2,1) = -1; bh(2) = -PhPrev + lim.ramp*dtv(1);
for k = 2:Nc
    Ah(2*k-1, k) = 1;  Ah(2*k-1, k-1) = -1; bh(2*k-1) = lim.ramp*dtv(min(k,H));
    Ah(2*k,   k) = -1; Ah(2*k,   k-1) = 1;  bh(2*k)   = lim.ramp*dtv(min(k,H));
end
lbh = zeros(Nc,1); ubh = lim.phMax*ones(Nc,1);

% Iteration-invariant part of pumped-storage subproblem
ns = 2*Nc + 2*H;                       % [ug; up; sp; sm]
jg = 1:Nc; jp = Nc+1:2*Nc; js1 = 2*Nc+1:2*Nc+H; js2 = 2*Nc+H+1:ns;
% g(z) = T*ug - T*up + sp - sm
Gmap = [T, -T, eye(H), -eye(H)];
Hs = 2*wdu*blkdiag(D'*D, D'*D, zeros(2*H)) + rho*(Gmap'*Gmap) + 1e-9*eye(ns);
fs0 = zeros(ns,1);
fs0(jg) = 18 * (T'*dtv);
fs0(jp) = T' * (price .* dtv);
fs0(js1) = 1000 * dtv;
fs0(js2) = 80 * dtv;
aE = zeros(ns,1);
aE(jg) = -T'*dtv / lim.etaG;
aE(jp) =  T'*dtv * lim.etaP;
Hs = Hs + 2*wE*(aE*aE');
fs0 = fs0 + 2*wE*(E0 - lim.eRef)*aE;
% Energy corridor of energy storage
As = zeros(2*H, ns); bs = zeros(2*H, 1);
for k = 1:H
    cg = zeros(1,Nc); cp = zeros(1,Nc);
    for i = 1:k
        cg = cg - dtv(i)/lim.etaG * T(i,:);
        cp = cp + dtv(i)*lim.etaP * T(i,:);
    end
    As(2*k-1, jg) = cg;  As(2*k-1, jp) = cp;  bs(2*k-1) = lim.eMax - E0;
    As(2*k,   jg) = -cg; As(2*k,   jp) = -cp; bs(2*k)   = E0;
end
lbs = zeros(ns,1); ubs = [lim.pgMax*ones(Nc,1); lim.ppMax*ones(Nc,1); inf(2*H,1)];

Ph = min(max(nFc, 0), lim.phMax);      % Initial hydropower trajectory
g = nFc - Ph;
ok = true;
priTrace = zeros(maxIter,1); duaTrace = zeros(maxIter,1);
for it = 1:maxIter
    % --- Hydropower subproblem ---
    v = nFc - g - u;
    fh = fh0 - rho*(T'*v);
    [zh, ~, flg1] = quadprog(Hh, fh, Ah, bh, [], [], lbh, ubh, [], opts);
    if isempty(zh) || flg1 <= 0, ok = false; zh = min(max(v(1:Nc),0),lim.phMax); end
    Ph = T * zh;

    % --- Pumped-storage subproblem (including reserve constraints; hydropower output passed in as parameters)---
    w = nFc - Ph - u;
    fs = fs0 - rho*(Gmap'*w);
    Ar = [As; [T, -T, zeros(H, 2*H)]];
    br = [bs; lim.phMax + lim.pgMax - 0.05*loadFc - Ph];
    gOld = g;
    [zs, ~, flg2] = quadprog(Hs, fs, Ar, br, [], [], lbs, ubs, [], opts);
    if isempty(zs) || flg2 <= 0, ok = false; zs = zeros(ns,1); end
    g = Gmap * zs;

    % --- Dual update and residuals ---
    r = Ph + g - nFc;
    u = u + r;
    priTrace(it) = norm(r);
    duaTrace(it) = rho * norm(g - gOld);
    if priTrace(it) < tolPri && duaTrace(it) < tolDua
        break;
    end
end

sol.Ph = Ph;
sol.Pg = T * zs(jg);
sol.Pp = T * zs(jp);
both = sol.Pg > 1 & sol.Pp > 1;
swap = sol.Pg >= sol.Pp;
sol.Pp(both & swap) = 0;
sol.Pg(both & ~swap) = 0;

info = struct('iters', it, 'rPri', priTrace(it), 'rDua', duaTrace(it), ...
    'priTrace', priTrace(1:it), 'duaTrace', duaTrace(1:it), 'ok', ok, 'tol', tolPri);
info.duals = u;
end
