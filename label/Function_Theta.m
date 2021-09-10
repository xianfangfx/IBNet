function y = Function_Theta(x)

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

%%
beta = 4/pi*atan(1/alpha);
y = 2/pi/beta*atan(2/alpha*(x-1/2))+1/2;
%%