%% 复现的原方法和所提的新方法在1张图片上的结果对比

clc, clear, close all;

%% %% %% 复现的原方法
%% %% Mask
%%
Mask = imread('Sample1.png');
% Mask = imread('Sample2.png');
% Mask = imread('Sample3.png');
% Mask = imread('Sample4.png');
% Mask = imread('Sample5.png');
% Mask = imread('Sample6.png');
% Mask = imread('Temp.png');
%%
Mask = im2double(Mask);  %% 前 (默认)
%%
fprintf('Mask中大于0的个数为:%d\n',length(Mask(Mask>0)));
%%
% fprintf('Mask的最小值是:%.2f\n',min(min(Mask)));
% fprintf('Mask的最大值是:%.2f\n',max(max(Mask)));
% figure;
% imshow(Mask)
% imwrite(Mask,'初始图.png');
%% %% Blur
%%
Blur = Mask;  %% 不模糊化 (默认)
%%
% Blur = imfilter(Mask,fspecial('average',[5 5]),'replicate');  %% 模糊化
%%
% fprintf('经过Blur后的最小值是:%.2f\n',min(min(Blur)));
% fprintf('经过Blur后的最大值是:%.2f\n',max(max(Blur)));
% figure;
% imshow(Blur); 
% imwrite(Blur,'经过模糊处理后.png');
%%
%% %% DistanceTransform
Blur = ~Blur;
DistTrans = bwdist(Blur,'euclidean');
%%
% fprintf('经过DistanceTransform后的最小值是:%.2f\n',min(min(DistTrans)));
% fprintf('经过DistanceTransform后的最大值是:%.2f\n',max(max(DistTrans)));
% figure;
% imshow(DistTrans); 
% imwrite(DistTrans,'再经过距离变换处理后.png');
%% %% ^0.5
% Power = DistTrans;
Power = DistTrans.^0.5;  %% (默认)
%%
% fprintf('经过^0.5后的最小值是:%.2f\n',min(min(Power)));
% fprintf('经过^0.5后的最大值是:%.2f\n',max(max(Power)));
% figure;
% imshow(Power); 
% imwrite(Power,'再再经过次方处理后.png');
%% %% Normalization
%%
Normalization = (Power-min(min(Power)))./(max(max(Power))-min(min(Power)));  %% (默认)
% Normalization = (Power-min(min(Power)))./(max(max(Power))-min(min(Power)))*255;
%%
fprintf('Normalization中大于0的个数为:%d\n',length(Normalization(Normalization>0)));
%%
% fprintf('经过Normalization后的最小值是:%.2f\n',min(min(Normalization)));
% fprintf('经过Normalization后的最大值是:%.2f\n',max(max(Normalization)));
% figure;
% imshow(Normalization); 
% imwrite(Normalization,'再再再经过归一化处理后.png');
%% %% 画图
%% 不鲁棒
% PreB = Normalization;
% PreD = Mask-Normalization;
% figure;
% subplot(121);
% imshow(PreB);
% title('PreB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% subplot(122);
% imshow(PreD);
% title('PreD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
%% 鲁棒 (默认)
B = Mask.*Normalization;
D = Mask-Mask.*Normalization;
figure;
subplot(121);
imshow(B);
title('B','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(B,'原-B.png');
subplot(122);
imshow(D);
title('D','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(D,'原-D.png');

%% %% %% 所提的新方法
ConvetSB = Normalization;
ConvetWB = Normalization;
for i = 1:size(Normalization,1)
    for j = 1:size(Normalization,2)
        %% 方法1
%          ConvetSB(i,j) = Function_Theta(Mask(i,j).*Normalization(i,j));  %% Strengthened
%          ConvetWB(i,j) = Function_Gamma(Mask(i,j).*Normalization(i,j));  %% Weakened
        %% 方法2
%          ConvetSB(i,j) = Function_Theta(Normalization(i,j));  %% Strengthened
%          ConvetWB(i,j) = Function_Gamma(Normalization(i,j));  %% Weakened
        %% 方法3 (默认)
         ConvetSB(i,j) = Normalization(i,j).*Function_Theta(Normalization(i,j));  %% Strengthened
         ConvetWB(i,j) = Normalization(i,j).*Function_Gamma(Normalization(i,j));  %% Weakened
    end
end
%%
% Mask = im2double(Mask);  %% 后
%%
ConvetSB = im2double(ConvetSB);
ConvetWB = im2double(ConvetWB);
%% %% 画图
%% 不鲁棒
% SB = ConvetSB;
% WB = ConvetWB;
% SD = Mask-ConvetSB;
% WD = Mask-ConvetWB;
% figure;
% subplot(221);
% imshow(SB);
% title('SB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% imwrite(SB,'新-不鲁棒SB.png');
% subplot(222);
% imshow(SD);
% title('SD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% imwrite(SD,'新-不鲁棒SD.png');
% subplot(223);
% imshow(WB);
% title('WB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% imwrite(WB,'新-不鲁棒WB.png');
% subplot(224);
% imshow(WD);
% title('WD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% imwrite(WD,'新-不鲁棒WD.png');
%% 鲁棒 (默认)
SB = Mask.*ConvetSB;
WB = Mask.*ConvetWB;
SD = Mask-Mask.*ConvetSB;
WD = Mask-Mask.*ConvetWB;
figure;
subplot(221);
imshow(SB);
title('SB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(SB,'新-鲁棒SB.png');
subplot(222);
imshow(SD);
title('SD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(SD,'新-鲁棒SD.png');
subplot(223);
imshow(WB);
title('WB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(WB,'新-鲁棒WB.png');
subplot(224);
imshow(WD);
title('WD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(WD,'新-鲁棒WD.png');