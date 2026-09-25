function S = sample_pairs(ds)
% Organize data into one-step transition samples, keep all points without filtering.
% State x = [hydropower, pumped-storage generation, pumping, net load]; exogenous u = [load, wind, solar, sin, cos].
% The first sample of each scenario group has no in-group predecessor; conservatively initialized with its net load and marked in firstFlag.

n = numel(ds.load);
X = zeros(n, 4);            % Previous time-step state
cur = [ds.hydro, ds.gen, ds.pump, ds.net];
firstFlag = false(n,1);
for i = 1:n
    if i > 1 && ds.group(i) == ds.group(i-1)
        X(i,:) = cur(i-1,:);
    else
        X(i,:) = [ds.net(i), 0, 0, ds.net(i)];
        firstFlag(i) = true;
    end
end

ang = 2*pi*(ds.hour + ds.minute/60) / 24;
S.X  = X;
S.U  = [ds.load, ds.wind, ds.pv, sin(ang), cos(ang)];
S.Y  = cur;
S.dt = ds.dt;
S.group = ds.group;
S.seq = ds.seq;
S.first = firstFlag;
S.n = n;
end
