clc, clear, close all;

%%
Mask = imread('Sample4.png'); 
% Mask = imread('Sample5.png'); 
% Mask = imread('Sample6.png'); 
%%

%%
B = imread('B.png');
D = imread('D.png');
SB = imread('SB.png');
WB = imread('WB.png');
SD = imread('SD.png');
WD = imread('WD.png');
%%

%%
f = figure; 
hax = axes(f); 
imagesc(hax, Mask);
set(gca, 'position', [0 0 1 1]);
Mask = getframe(hax);
imwrite(Mask.cdata, 'Mask_new.png');
Mask = imread('Mask_new.png');
Mask = imresize(Mask, [300, 400]);
imwrite(Mask, 'Mask_new.png');
%%

%%
f = figure; 
hax = axes(f); 
imagesc(hax, B);
set(gca, 'position', [0 0 1 1]);
B = getframe(hax);
imwrite(B.cdata, 'B_new.png');
B = imread('B_new.png');
B = imresize(B, [300, 400]);
imwrite(B, 'B_new.png');
%%

%%
f = figure; 
hax = axes(f); 
imagesc(hax, D);
set(gca, 'position', [0 0 1 1]);
D = getframe(hax);
imwrite(D.cdata, 'D_new.png');
D = imread('D_new.png');
D = imresize(D, [300, 400]);
imwrite(D, 'D_new.png');
%%

%%
f = figure; 
hax = axes(f); 
imagesc(hax, SB);
set(gca, 'position', [0 0 1 1]);
SB = getframe(hax);
imwrite(SB.cdata, 'SB_new.png');
SB = imread('SB_new.png');
SB = imresize(SB, [300, 400]);
imwrite(SB, 'SB_new.png');
%%

%%
f = figure; 
hax = axes(f); 
imagesc(hax, WB);
set(gca, 'position', [0 0 1 1]);
WB = getframe(hax);
imwrite(WB.cdata, 'WB_new.png');
WB = imread('WB_new.png');
WB = imresize(WB, [300, 400]);
imwrite(WB, 'WB_new.png');
%%

%%
f = figure; 
hax = axes(f); 
imagesc(hax, SD);
set(gca, 'position', [0 0 1 1]);
SD = getframe(hax);
imwrite(SD.cdata, 'SD_new.png');
SD = imread('SD_new.png');
SD = imresize(SD, [300, 400]);
imwrite(SD, 'SD_new.png');
%%

%%
f = figure; 
hax = axes(f); 
imagesc(hax, WD);
set(gca, 'position', [0 0 1 1]);
WD = getframe(hax);
imwrite(WD.cdata, 'WD_new.png');
WD = imread('WD_new.png');
WD = imresize(WD, [300, 400]);
imwrite(WD, 'WD_new.png');
%%