%% ËùÌáµÄGamma(x)

function y = Function_Gamma(x)

%%
% alpha = 0.000001;
% alpha = 0.01;
% alpha = 0.05;
% alpha = 0.1;
% alpha = 0.2;
alpha = 0.3; 
% alpha = 0.4;
% alpha = 0.5;
% alpha = 1;
% alpha = 999999;
%%

beta = 4/pi*atan(1/alpha);
y = alpha/2*tan(pi*beta/2*(x-1/2))+1/2;