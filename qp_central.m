function [sol, info] = qp_central(nFc, dtv, price, PhPrev, E0, loadFc, Np, Nc, lim) %#ok<INUSD>
% Centralized rolling horizon optimization: a single convex QP jointly optimizes hydropower, pumped-storage and slack variables.
% Objective = operational cost + under/over-supply penalty + control increment smoothing + terminal energy state deviation;
% Constraints = power balance (with slack), equipment limits, hydropower ramping, stored energy, spinning reserve.
% Mutual exclusivity between generation and pumping is enforced via round-trip efficiency loss and post-solution mode correction.

H = numel(nFc);
Nc = min(Nc, H);
[T, D] = horizon_maps(H, Nc);
nv = 3*Nc + 2*H;                 % [uh; ug; up; sp; sm]
ih = 1:Nc; ig = Nc+1:2*Nc; ip = 2*Nc+1:3*Nc;
is1 = 3*Nc+1:3*Nc+H; is2 = 3*Nc+H+1:nv;

wdu = 1e-4; wE = 2e-4;

Hq = 1e-8 * eye(nv);
Hq(ih,ih) = Hq(ih,ih) + 2*wdu*(D'*D);
Hq(ig,ig) = Hq(ig,ig) + 2*wdu*(D'*D);
Hq(ip,ip) = Hq(ip,ip) + 2*wdu*(D'*D);

f = zeros(nv,1);
f(ih) = T' * (11*dtv);
f(ig) = T' * (18*dtv);
f(ip) = T' * (price .* dtv);
f(is1) = 1000 * dtv;
f(is2) = 80 * dtv;
% The first increment term is defined relative to the actual hydropower output at the previous time step.
f(ih(1)) = f(ih(1)) - 2*wdu*PhPrev;
Hq(ih(1),ih(1)) = Hq(ih(1),ih(1)) + 2*wdu;

% Terminal stored energy pulled back to reference value (soft constraint).
aE = zeros(nv,1);
aE(ig) = -T'*dtv / lim.etaG;
aE(ip) =  T'*dtv * lim.etaP;
Hq = Hq + 2*wE*(aE*aE');
f = f + 2*wE*(E0 - lim.eRef)*aE;

% Power balance equality with bidirectional slack variables.
Aeq = [T, T, -T, eye(H), -eye(H)];
beq = nFc;

[Ain, bin] = shared_ineq(T, dtv, PhPrev, E0, loadFc, Nc, H, lim, nv, ih, ig, ip);

lb = zeros(nv,1);
ub = [lim.phMax*ones(Nc,1); lim.pgMax*ones(Nc,1); lim.ppMax*ones(Nc,1); inf(2*H,1)];

opts = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
[z, ~, flag] = quadprog(Hq, f, Ain, bin, Aeq, beq, lb, ub, [], opts);
if isempty(z) || flag <= 0
    % Fall back to tracking-based safe solution if solver fails, and report the status truthfully.
    z = zeros(nv,1);
    z(ih) = min(max(nFc(1:Nc), 0), lim.phMax);
    info.ok = false;
else
    info.ok = true;
end
info.flag = flag;

sol.Ph = T * z(ih);
sol.Pg = T * z(ig);
sol.Pp = T * z(ip);
% Mutual-exclusivity correction: simultaneous activation rarely occurs under round-trip loss, but explicitly handled for robustness.
both = sol.Pg > 1 & sol.Pp > 1;
swap = sol.Pg >= sol.Pp;
sol.Pp(both & swap) = 0;
sol.Pg(both & ~swap) = 0;
end
