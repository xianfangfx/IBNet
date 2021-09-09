%% 所提的Gamma(x)和Theta(x)的可视化
%% 3条实线：0.5、0.3、0.1

clc, clear, close all;

a = 0;
b = 1;
x = a:0.0001:b;

%% %% Theta(x)
%% alpha = 999999
figure;
alpha = 999999;
beta = 4/pi*atan(1/alpha);
y = 2/beta/pi*atan(2/alpha*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f1 = plot(x,y,'k--');
%% alpha = 0.5
hold on;
alpha = 0.5;
beta = 4/pi*atan(1/alpha);
y = 2/beta/pi*atan(2/alpha*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f2 = plot(x,y,'c-');
%% alpha = 0.3
hold on;
alpha = 0.3;
beta = 4/pi*atan(1/alpha);
y = 2/beta/pi*atan(2/alpha*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f3 = plot(x,y,'b-');
%% alpha = 0.1
hold on;
alpha = 0.1;
beta = 4/pi*atan(1/alpha);
y = 2/beta/pi*atan(2/alpha*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f4 = plot(x,y,'m-');
%% alpha = 0.000001
hold on
alpha = 0.000001;
beta = 4/pi*atan(1/alpha);
y = 2/beta/pi*atan(2/alpha*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f5 = plot(x,y,'r--');
%
h = legend([f1,f2,f3,f4,f5],'$\alpha\rightarrow\infty$','$\alpha=0.5$','$\alpha=0.3$','$\alpha=0.1$','$\alpha\rightarrow0$');
set(h,'Interpreter','latex','FontName','Times New Roman','FontSize',10);
xlabel('$x$','Interpreter','latex','FontName','Times New Roman','FontSize',10);
ylabel('$\Theta(x)$','Interpreter','latex','FontName','Times New Roman','FontSize',10);
axis([0,1,0,1]);
box on;
grid on;

%% %% Gamma(x)
%% alpha = 999999
figure;
alpha = 999999;
beta = 4/pi*atan(1/alpha);
y = alpha/2*tan(pi*beta/2*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f1 = plot(x,y,'k--');
%% alpha = 0.5
hold on;
alpha = 0.5;
beta = 4/pi*atan(1/alpha);
y = alpha/2*tan(pi*beta/2*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f2 = plot(x,y,'c-');
%% alpha = 0.3
hold on;
alpha = 0.3;
beta = 4/pi*atan(1/alpha);
y = alpha/2*tan(pi*beta/2*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f3 = plot(x,y,'b-');
%% alpha = 0.1
hold on;
alpha = 0.1;
beta = 4/pi*atan(1/alpha);
y = alpha/2*tan(pi*beta/2*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f4 = plot(x,y,'m-');
%% alpha = 0.000001
hold on
alpha = 0.000001;
beta = 4/pi*atan(1/alpha);
y = alpha/2*tan(pi*beta/2*(x-1/2))+1/2;
temp = (b-a)./(max(y)-min(y));
y = a+temp*(y-min(y));
f5 = plot(x,y,'r--');
%
h = legend([f1,f2,f3,f4,f5],'$\alpha\rightarrow\infty$','$\alpha=0.5$','$\alpha=0.3$','$\alpha=0.1$','$\alpha\rightarrow0$');
set(h,'Interpreter','latex','FontName','Times New Roman','FontSize',10);
xlabel('$x$','Interpreter','latex','FontName','Times New Roman','FontSize',10);
ylabel('$\Gamma(x)$','Interpreter','latex','FontName','Times New Roman','FontSize',10);
axis([0,1,0,1]);
box on;
grid on;