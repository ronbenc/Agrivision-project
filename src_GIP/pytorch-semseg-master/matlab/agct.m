function[C, minC, maxC] = agct(RGB, N, alpha_params, EPS, MAX_VAL)

if ~exist('EPS','var')
    EPS = 1e-3;
end
if ~exist('MAX_VAL','var')
    MAX_VAL = 1e3;
end
a = alpha_params;
R = RGB(:,:,1);
G = RGB(:,:,2);
B = RGB(:,:,3);

nomin = a(1)*R + a(2)*G + a(3)*B + a(4)*N + a(5);
denom = a(6)*R + a(7)*G + a(8)*B + a(9)*N + a(10);
denom(denom == 0.0) = EPS;
C = nomin./denom;
% C(isnan(C)) = 0.0;
% C((isinf(C) .* sign(C)) > 0) = MAX_VAL;
% C((isinf(C) .* sign(C)) < 0) = -MAX_VAL;
minC = min(C(:));
maxC = max(C(:));

return

