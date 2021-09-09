%% ������·�����10553��ͼƬ�ϵĽ��

clc, clear, close all;

tic;
DataDir = fullfile('Data\DUTS-TR\Mask\');
OtpDir1 = fullfile('Output\DUTS-TR\body1-origin\');
OtpDir2 = fullfile('Output\DUTS-TR\body2-origin\');
OtpDir3 = fullfile('Output\DUTS-TR\detail1-origin\');
OtpDir4 = fullfile('Output\DUTS-TR\detail2-origin\');
Dir = dir(fullfile(DataDir,'*.png'));
FileNames = {Dir.name};
if ~exist(OtpDir1,'dir')
    mkdir(OtpDir1)
end
if ~exist(OtpDir2,'dir')
    mkdir(OtpDir2)
end
if ~exist(OtpDir3,'dir')
    mkdir(OtpDir3)
end
if ~exist(OtpDir4,'dir')
    mkdir(OtpDir4)
end
for k = 1:1:10553
    FileName = FileNames{1,k};
    FigureName = strcat(DataDir,FileName);
    Mask = imread(FigureName);
    %%
    Mask = im2double(Mask);  %% ǰ (Ĭ��)
   %%
    filename = FileName;
    path1 = fullfile(OtpDir1,filename);
    path2 = fullfile(OtpDir2,filename);
    path3 = fullfile(OtpDir3,filename);
    path4 = fullfile(OtpDir4,filename);
    %%
    Blur = Mask;  %% ��ģ���� (Ĭ��)
    %%
%     Blur = imfilter(Mask,fspecial('average',[5 5]),'replicate');  %% ģ����
    %%
    Blur = ~Blur;
    DistTrans = bwdist(Blur,'euclidean');
    Power = DistTrans.^0.5;
    Normalization = (Power-min(min(Power)))./(max(max(Power))-min(min(Power)));
    ConvertSB = Normalization;
    ConvertWB = Normalization;
    for i = 1:size(Normalization,1)
        for j = 1:size(Normalization,2)
           %% ����1
%             ConvertSB(i,j) = Function_Theta(Mask(i,j).*Normalization(i,j));  %% Strengthened
%             ConvertWB(i,j) = Function_Gamma(Mask(i,j).*Normalization(i,j));  %% Weakened
           %% ����2
%             ConvertSB(i,j) = Function_Theta(Normalization(i,j));  %% Strengthened
%             ConvertWB(i,j) = Function_Gamma(Normalization(i,j));  %% Weakened
           %% ����3 (Ĭ��)
            ConvertSB(i,j) = Normalization(i,j).*Function_Theta(Normalization(i,j));  %% Strengthened
            ConvertWB(i,j) = Normalization(i,j).*Function_Gamma(Normalization(i,j));  %% Weakened
        end
    end
    %%
%     Mask = im2double(Mask);  %% ��
    %%
    ConvertSB = im2double(ConvertSB);
    ConvertWB = im2double(ConvertWB);
    %% ��³��
%     SB = ConvertSB;
%     WB = ConvertWB;
%     SD = Mask-ConvertSB;
%     WD = Mask-ConvertWB;
%     imwrite(SB,path1);
%     imwrite(WB,path2);
%     imwrite(SD,path3);
%     imwrite(WD,path4);
    %% ³�� (Ĭ��)
    SB = Mask.*ConvertSB;
    WB = Mask.*ConvertWB;
    SD = Mask-Mask.*ConvertSB;
    WD = Mask-Mask.*ConvertWB;
    imwrite(SB,path1);
    imwrite(WB,path2);
    imwrite(SD,path3);
    imwrite(WD,path4);
end
toc;