%% ���ֵ�ԭ������������·�����1��ͼƬ�ϵĽ���Ա�

clc, clear, close all;

%% %% %% ���ֵ�ԭ����
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
Mask = im2double(Mask);  %% ǰ (Ĭ��)
%%
fprintf('Mask�д���0�ĸ���Ϊ:%d\n',length(Mask(Mask>0)));
%%
% fprintf('Mask����Сֵ��:%.2f\n',min(min(Mask)));
% fprintf('Mask�����ֵ��:%.2f\n',max(max(Mask)));
% figure;
% imshow(Mask)
% imwrite(Mask,'��ʼͼ.png');
%% %% Blur
%%
Blur = Mask;  %% ��ģ���� (Ĭ��)
%%
% Blur = imfilter(Mask,fspecial('average',[5 5]),'replicate');  %% ģ����
%%
% fprintf('����Blur�����Сֵ��:%.2f\n',min(min(Blur)));
% fprintf('����Blur������ֵ��:%.2f\n',max(max(Blur)));
% figure;
% imshow(Blur); 
% imwrite(Blur,'����ģ�������.png');
%%
%% %% DistanceTransform
Blur = ~Blur;
DistTrans = bwdist(Blur,'euclidean');
%%
% fprintf('����DistanceTransform�����Сֵ��:%.2f\n',min(min(DistTrans)));
% fprintf('����DistanceTransform������ֵ��:%.2f\n',max(max(DistTrans)));
% figure;
% imshow(DistTrans); 
% imwrite(DistTrans,'�پ�������任�����.png');
%% %% ^0.5
% Power = DistTrans;
Power = DistTrans.^0.5;  %% (Ĭ��)
%%
% fprintf('����^0.5�����Сֵ��:%.2f\n',min(min(Power)));
% fprintf('����^0.5������ֵ��:%.2f\n',max(max(Power)));
% figure;
% imshow(Power); 
% imwrite(Power,'���پ����η������.png');
%% %% Normalization
%%
Normalization = (Power-min(min(Power)))./(max(max(Power))-min(min(Power)));  %% (Ĭ��)
% Normalization = (Power-min(min(Power)))./(max(max(Power))-min(min(Power)))*255;
%%
fprintf('Normalization�д���0�ĸ���Ϊ:%d\n',length(Normalization(Normalization>0)));
%%
% fprintf('����Normalization�����Сֵ��:%.2f\n',min(min(Normalization)));
% fprintf('����Normalization������ֵ��:%.2f\n',max(max(Normalization)));
% figure;
% imshow(Normalization); 
% imwrite(Normalization,'�����پ�����һ�������.png');
%% %% ��ͼ
%% ��³��
% PreB = Normalization;
% PreD = Mask-Normalization;
% figure;
% subplot(121);
% imshow(PreB);
% title('PreB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% subplot(122);
% imshow(PreD);
% title('PreD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
%% ³�� (Ĭ��)
B = Mask.*Normalization;
D = Mask-Mask.*Normalization;
figure;
subplot(121);
imshow(B);
title('B','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(B,'ԭ-B.png');
subplot(122);
imshow(D);
title('D','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(D,'ԭ-D.png');

%% %% %% ������·���
ConvetSB = Normalization;
ConvetWB = Normalization;
for i = 1:size(Normalization,1)
    for j = 1:size(Normalization,2)
        %% ����1
%          ConvetSB(i,j) = Function_Theta(Mask(i,j).*Normalization(i,j));  %% Strengthened
%          ConvetWB(i,j) = Function_Gamma(Mask(i,j).*Normalization(i,j));  %% Weakened
        %% ����2
%          ConvetSB(i,j) = Function_Theta(Normalization(i,j));  %% Strengthened
%          ConvetWB(i,j) = Function_Gamma(Normalization(i,j));  %% Weakened
        %% ����3 (Ĭ��)
         ConvetSB(i,j) = Normalization(i,j).*Function_Theta(Normalization(i,j));  %% Strengthened
         ConvetWB(i,j) = Normalization(i,j).*Function_Gamma(Normalization(i,j));  %% Weakened
    end
end
%%
% Mask = im2double(Mask);  %% ��
%%
ConvetSB = im2double(ConvetSB);
ConvetWB = im2double(ConvetWB);
%% %% ��ͼ
%% ��³��
% SB = ConvetSB;
% WB = ConvetWB;
% SD = Mask-ConvetSB;
% WD = Mask-ConvetWB;
% figure;
% subplot(221);
% imshow(SB);
% title('SB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% imwrite(SB,'��-��³��SB.png');
% subplot(222);
% imshow(SD);
% title('SD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% imwrite(SD,'��-��³��SD.png');
% subplot(223);
% imshow(WB);
% title('WB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% imwrite(WB,'��-��³��WB.png');
% subplot(224);
% imshow(WD);
% title('WD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
% imwrite(WD,'��-��³��WD.png');
%% ³�� (Ĭ��)
SB = Mask.*ConvetSB;
WB = Mask.*ConvetWB;
SD = Mask-Mask.*ConvetSB;
WD = Mask-Mask.*ConvetWB;
figure;
subplot(221);
imshow(SB);
title('SB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(SB,'��-³��SB.png');
subplot(222);
imshow(SD);
title('SD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(SD,'��-³��SD.png');
subplot(223);
imshow(WB);
title('WB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(WB,'��-³��WB.png');
subplot(224);
imshow(WD);
title('WD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(WD,'��-³��WD.png');