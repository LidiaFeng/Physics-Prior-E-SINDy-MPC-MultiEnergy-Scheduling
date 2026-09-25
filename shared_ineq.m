function [Ain, bin] = shared_ineq(T, dtv, PhPrev, E0, loadFc, Nc, H, lim, nv, ih, ig, ip)
% Inequality constraints for centralized QP: hydropower ramping, energy bounds of energy storage, spinning reserve.

rows = {};
rhs = {};

% Hydropower ramping (including first step relative to previous actual output)
r1 = zeros(2*Nc, nv); b1 = zeros(2*Nc, 1);
r1(1, ih(1)) = 1;  b1(1) = PhPrev + lim.ramp*dtv(1);
r1(2, ih(1)) = -1; b1(2) = -PhPrev + lim.ramp*dtv(1);
for k = 2:Nc
    r1(2*k-1, ih(k)) = 1;  r1(2*k-1, ih(k-1)) = -1; b1(2*k-1) = lim.ramp*dtv(min(k,H));
    r1(2*k,   ih(k)) = -1; r1(2*k,   ih(k-1)) = 1;  b1(2*k)   = lim.ramp*dtv(min(k,H));
end
rows{end+1} = r1; rhs{end+1} = b1;

% Storage energy corridor 0 <= E_k <= eMax
% E_k = E0 + sum_i<=k dtv_i*(etaP*Pp_i - Pg_i/etaG)
r2 = zeros(2*H, nv); b2 = zeros(2*H, 1);
for k = 1:H
    cg = zeros(1, Nc); cp = zeros(1, Nc);
    for i = 1:k
        cg = cg - dtv(i)/lim.etaG * T(i,:);
        cp = cp + dtv(i)*lim.etaP * T(i,:);
    end
    r2(2*k-1, ig) = cg;  r2(2*k-1, ip) = cp;  b2(2*k-1) = lim.eMax - E0;   % E_k <= eMax
    r2(2*k,   ig) = -cg; r2(2*k,   ip) = -cp; b2(2*k)   = E0;              % E_k >= 0
end
rows{end+1} = r2; rhs{end+1} = b2;

% Spinning reserve: (phMax-Ph)+(pgMax-Pg)+Pp >= 5% of load.
r3 = zeros(H, nv); b3 = zeros(H, 1);
for k = 1:H
    r3(k, ih) = T(k,:); r3(k, ig) = T(k,:); r3(k, ip) = -T(k,:);
    b3(k) = lim.phMax + lim.pgMax - 0.05*loadFc(k);
end
rows{end+1} = r3; rhs{end+1} = b3;

Ain = cell2mat(rows');
bin = cell2mat(rhs');
end
