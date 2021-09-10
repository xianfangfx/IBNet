clc, clear, close all;

%%
% Mask = imread('Sample1.png');
% Mask = imread('Sample2.png');
% Mask = imread('Sample3.png');
Mask = imread('Sample4.png');
% Mask = imread('Sample5.png');
% Mask = imread('Sample6.png');
% Mask = imread('Temp.png');
%%

%%
imwrite(Mask,'Mask.png');
Mask = im2double(Mask);
Blur = Mask;
Blur = ~Blur;
DistTrans = bwdist(Blur,'euclidean');
Power = DistTrans.^0.5;
Normalization = (Power-min(min(Power)))./(max(max(Power))-min(min(Power)));
%%

%%
B = Mask.*Normalization;
D = Mask-Mask.*Normalization;
figure;
subplot(121);
imshow(B);
title('B','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(B,'B.png');
subplot(122);
imshow(D);
title('D','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(D,'D.png');
%%

%%
ConvetSB = Normalization;
ConvetWB = Normalization;
for i = 1:size(Normalization,1)
    for j = 1:size(Normalization,2)
         ConvetSB(i,j) = Normalization(i,j).*Function_Theta(Normalization(i,j));
         ConvetWB(i,j) = Normalization(i,j).*Function_Gamma(Normalization(i,j));
    end
end
ConvetSB = im2double(ConvetSB);
ConvetWB = im2double(ConvetWB);
SB = Mask.*ConvetSB;
WB = Mask.*ConvetWB;
SD = Mask-Mask.*ConvetSB;
WD = Mask-Mask.*ConvetWB;
figure;
subplot(221);
imshow(SB);
title('SB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(SB,'SB.png');
subplot(222);
imshow(SD);
title('SD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(SD,'SD.png');
subplot(223);
imshow(WB);
title('WB','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(WB,'WB.png');
subplot(224);
imshow(WD);
title('WD','Interpreter','latex','FontName','Times New Roman','FontSize',10);
imwrite(WD,'WD.png');
%%