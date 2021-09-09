clc, clear, close all;

%%
Mask = imread('Sample4.png'); 
% Mask = imread('Sample5.png'); 
% Mask = imread('Sample6.png'); 
%%
B = imread('原-B.png');
D = imread('原-D.png');
SB = imread('新-鲁棒SB.png');
WB = imread('新-鲁棒WB.png');
SD = imread('新-鲁棒SD.png');
WD = imread('新-鲁棒WD.png');

f = figure; 
hax = axes(f); 
imagesc(hax, Mask);
set(gca, 'position', [0 0 1 1]);
Mask = getframe(hax);
imwrite(Mask.cdata, 'Mask.png');
Mask = imread('Mask.png');
Mask = imresize(Mask, [300, 400]);
imwrite(Mask, 'Mask.png');

f = figure; 
hax = axes(f); 
imagesc(hax, B);
set(gca, 'position', [0 0 1 1]);
B = getframe(hax);
imwrite(B.cdata, 'B.png');
B = imread('B.png');
B = imresize(B, [300, 400]);
imwrite(B, 'B.png');

f = figure; 
hax = axes(f); 
imagesc(hax, D);
set(gca, 'position', [0 0 1 1]);
D = getframe(hax);
imwrite(D.cdata, 'D.png');
D = imread('D.png');
D = imresize(D, [300, 400]);
imwrite(D, 'D.png');

f = figure; 
hax = axes(f); 
imagesc(hax, SB);
set(gca, 'position', [0 0 1 1]);
SB = getframe(hax);
imwrite(SB.cdata, 'SB.png');
SB = imread('SB.png');
SB = imresize(SB, [300, 400]);
imwrite(SB, 'SB.png');

f = figure; 
hax = axes(f); 
imagesc(hax, WB);
set(gca, 'position', [0 0 1 1]);
WB = getframe(hax);
imwrite(WB.cdata, 'WB.png');
WB = imread('WB.png');
WB = imresize(WB, [300, 400]);
imwrite(WB, 'WB.png');

f = figure; 
hax = axes(f); 
imagesc(hax, SD);
set(gca, 'position', [0 0 1 1]);
SD = getframe(hax);
imwrite(SD.cdata, 'SD.png');
SD = imread('SD.png');
SD = imresize(SD, [300, 400]);
imwrite(SD, 'SD.png');

f = figure; 
hax = axes(f); 
imagesc(hax, WD);
set(gca, 'position', [0 0 1 1]);
WD = getframe(hax);
imwrite(WD.cdata, 'WD.png');
WD = imread('WD.png');
WD = imresize(WD, [300, 400]);
imwrite(WD, 'WD.png');