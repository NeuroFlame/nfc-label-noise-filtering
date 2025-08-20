%% LAMP--20240731
% This code corresponds to the article titled "More reliable biomarkers and
% more accurate prediction for mental disorders using a label-noise filtering-based 
% dimensional prediction method" published in the journal iScience in 2024. 
% The copyright belongs to the Intelligent Analysis of Medical Images (IAMI) laboratory. 
% If you encounter any problems or need further communication during use, please contact us at duyuhui@sxu.edu.cn and sxxying@126.com. 
%% It is worth mentioning that this code should run on two or more datasets, and the more datasets, the more stable the results.
% Format of the input data: matrix, where each row represents a single sample, and each column represents a feature. 

clear
clc
close all

%% INPUT:
% Dataset Name, storage location of data and results
DataName = {'FBIRN', 'MPRC', 'BSNIP','COBRE'}; % Dataset Name
LoadDataPath = '..\data\'; % Data storage location
SavePath = '..\result\'; % Results storage location

% Parameters for CRF-based model
SamplingThs = 0.7; % Balance data - Downsampling ratio
iter = 101; % Maximum number of iterations, number of CRDT
ntree = 201; % Number of trees in CRDT
NI_threshold = 2; %  Noise intensity (NI)
% Parameters for selecting typical subjects
TypThs = 0.8;

%% Estimating the non-noise rate of each sample in each source dataset
CountNonNoise(DataName,LoadDataPath,SavePath,SamplingThs,iter,ntree,NI_threshold)

%% Finding typical subjects for each source dataset
FindTyp(DataName,LoadDataPath,SavePath,TypThs)

%% Predicting scores for each independent dataset
% Any one dataset as an independent dataset, with the remaining datasets serving as the source datasets
PredictScore(DataName,LoadDataPath,SavePath,TypThs)


%% Estimate the non-noise rate of each sample in each dataset
function [] = CountNonNoise(DataName,LoadDataPath,SavePath,SamplingThs,iter,ntree,NI_threshold)
for D = 1:length(DataName)
    load(strcat(LoadDataPath,DataName{D},'.mat'));
    data = eval(DataName{D});

    %% data preparation
    [OrginSampNum,OrginColNum] = size(data);
    OriLabel = data(:,end);
    ClassNum = length(unique(OriLabel)); % Number of classes
    SampleOriIndex = cell(ClassNum,1);
    for c = 1:ClassNum % Downsampling
        SampleOriIndex{c} = find(data(:,end)==c);
        SamplingNum(c) = floor(length(SampleOriIndex{c})*SamplingThs); 
    end

    % Counting: [[Sample ID] [number of times the sample was sampled]
    % [number of times the sample is non-noise] [The non-noise rate of the sample]]
    count = zeros(OrginSampNum,4);
    count(:,1) = [1:OrginSampNum]'; % Sample ID
    %% find nonNoise
    for t=1:iter
        fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Constructing No. %s CRF for %s dataset...\n',num2str(t),DataName{D});
        IndexTemp = [];
        for c = 1:ClassNum
            IndexTemp = [IndexTemp;SampleOriIndex{c}(randperm(length(SampleOriIndex{c}),floor(mean(SamplingNum))))];
        end
        SamplingIndex(:,t) = IndexTemp;
        count(SamplingIndex(:,t),2) = count(SamplingIndex(:,t),2)+1;
        Sample = data(SamplingIndex(:,t),:) ;
        Attr = zscore(Sample(:,1:OrginColNum-1));
        Label = Sample(:,OrginColNum);

        traindata = [Label Attr];
        [~,nonNoiseID,NLTCLabelS{t}]=ImprovedCRF_V1(traindata,ntree,NI_threshold); % identify non-noise subjects
        nonNoiseDataInd{t} = SamplingIndex(nonNoiseID,t); % ID of non-noise subjects

    end
    deNoiseCheck = zeros(size(data,1),1); 
    %%
    for i = 1:iter
        deNoiseCheck(nonNoiseDataInd{:,i}) = deNoiseCheck(nonNoiseDataInd{:,i})+1;
    end

    % 311 subject, for 100 iters , 70%, 201
    % sub i, no of smaple, non- noise
    deNoiseIndicator = deNoiseCheck./count(:,2);
    count(:,3) = deNoiseCheck;
    count(:,4) = deNoiseIndicator;
    save(strcat(SavePath,DataName{D},'_Count.mat'),'count','NLTCLabelS','nonNoiseDataInd')
    clear SamplingIndex
end
end

%% Find typical subjects for each dataset
function[] = FindTyp(DataName,LoadDataPath,SavePath,TypThs)
for D = 1:length(DataName)
    fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Selecting typical subjects for %s dataset...\n',DataName{D});

    load(strcat(SavePath,DataName{D},'_Count.mat'),'count')
    load(strcat(LoadDataPath,DataName{D},'.mat'));
    data = eval(DataName{D});
    OriLabel = data(:,size(data,2));
    clear FBIRN MPRC BSNIP COBRE
    TypIDG1 = find(count(:,4)>=TypThs & OriLabel==1);
    TypIDG2 = find(count(:,4)>=TypThs & OriLabel==2);
    save(strcat(SavePath,DataName{D},'_Typ.mat'),'TypIDG1','TypIDG2')

end
end

%% Predict scores for each independent dataset
function [] = PredictScore(DataName,LoadDataPath,SavePath,TypThs)
for IndID = 1:length(DataName) % FBIRN
    fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Scoring independent subjects in %s dataset...\n',DataName{IndID});
    load(strcat(LoadDataPath,DataName{IndID},'.mat'));
    IndepData = eval(DataName{IndID}); clear FBIRN MPRC BSNIP COBRE
    [SubNum, Col] = size(IndepData);
    IndepLabel = IndepData(:,Col);
    IndepScore = zeros(SubNum,length(DataName)+1); % The fifth column stores the mean scores of multiple groups.
    for MainID = 1:length(DataName) % Predicted scores for independent subjects based on typical subjects in each source dataset
        if IndID==MainID
            IndepScore(:,MainID) = IndepLabel;
            continue
        end % COBRE

        load(strcat(LoadDataPath,DataName{MainID},'.mat')); % load source dataset
        load(strcat(SavePath,DataName{MainID},'_Count.mat'),'count') % non-noise ratio for subjects in the source dataset

        data = eval(DataName{MainID}); clear FBIRN MPRC BSNIP COBRE
        TypData = data(count(:,4)>=TypThs,:); % typical subjects in the source dataset
        [IndepScore(:,MainID)] = ScoreComput(IndepData,TypData);
    end
    IndepScore(:,5) = mean(IndepScore(:,setdiff([1:length(DataName)],IndID)),2);
    idx = any(IndepScore(:,1:length(DataName))==0,2);
    IndepScore(idx,5) = 0;
    save(strcat(SavePath,DataName{IndID},'_Score.mat'),'IndepScore')
    clear IndepData IndepLabel TypData data count IndepScore idx
end
end

%% Predict scores for subjects in the independent dataset based on the typical subjects in the source datasets
function  [Scores] = ScoreComput(IndepData,MainData)
Col = size(IndepData,2);
IndepAttr = IndepData(:,1:Col-1);

MainAttr = MainData(:,1:Col-1);
MainLabel = MainData(:,Col);
for i = 1:length(unique(MainLabel))
    eval(['MainGroup',num2str(i),'=','MainAttr(MainLabel==i,:)',';']);
end
% Find features with significant inter-group differences
[~,Pval] = ttest2(MainGroup1,MainGroup2);
Fea = CumFea(Pval,0.01/(Col-1));

[Center1] = mean(MainGroup1(:,Fea));
[Center2] = mean(MainGroup2(:,Fea));

DisTypicalGroup1 = mean(pdist2(IndepAttr(:,Fea),Center1),2);
DisTypicalGroup2 = mean(pdist2(IndepAttr(:,Fea),Center2),2);
Distance = DisTypicalGroup1+DisTypicalGroup2;
A = DisTypicalGroup1./(Distance);
B = DisTypicalGroup2./(Distance);
Scores = tan((A-B)*pi/2);

end

%% Features with significant inter-group differences
% Pval: p-value for each feature [1*M]
% PvalPara: confidence threshold
function Fea = CumFea(Pval,PvalPara)
[SortPval, FeaInd] = sort(Pval);
Ind = find(SortPval>PvalPara);
Fea = FeaInd(1:Ind-1);
end