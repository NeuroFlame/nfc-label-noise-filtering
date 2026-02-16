%% Function for LAMP---20240731
% This code corresponds to the article titled "More reliable biomarkers and
% more accurate prediction for mental disorders using a label-noise filtering-based 
% dimensional prediction method" published in the journal iScience in 2024. 
% The copyright belongs to the Intelligent Analysis of Medical Images (IAMI) laboratory. 
% If you encounter any problems or need further communication during use, please contact us at duyuhui@sxu.edu.cn and sxxying@126.com. 

%% This function code draws on the code foundation provided by Xia in the article 
% "Complete random forest based class noise filtering learning for improving the generalizability of classifiers".
% Cite: Xia, S., Wang, G., Chen, Z., Duan, Y., and liu, Q. (2019). 
%       Complete random forest based class noise filtering learning for improving the generalizability of classifiers. 
%       IEEE Trans. Knowl. Data Eng. 31, 2063�C2078. https://doi.org/10.1109/TKDE.2018.2873791
% Input:
% train data([label attr]); ntree:Number of trees in CRDT; NI_threshold:Noise intensity (NI)
% Output:
% deNoiseData

function [ deNoiseData,nonNoiseID, NLTCLabelS]=ImprovedCRF_V1(traindata,ntree,NI_threshold)
% whether continuous features
isContiData=isContiFunc(traindata(:,2:end));
% Add sample ID in the laast column
traindata=addOrdNum(traindata);

%% construct CRF
simpRF1=cell(1,ntree);
simpRF2=cell(1,ntree);
nt = 2;
for ntree=1:ntree
    simpRF1{ntree}=simpRadTree(traindata,isContiData,[],1);
    simpRF2{ntree}=simpRadTree(traindata,isContiData,[],2);
end

traindata(:,end)=[];
numTrainSample=size(traindata,1);

%% traverse tree
% Each column presents a tree;
% Each row presents a sample;
% Each element represents the number of occurrences between the last two changes in the corresponding historical label sequence.
NLTCLabelS1(numTrainSample,ntree)=0;
NLTCLabelS2(numTrainSample,ntree)=0;
for nTree=1:ntree
    subresult=checkTreeLeaf(simpRF1{nTree});
    subresult=sortrows(subresult,1); % Sort
    NLTCLabelS1(:,nTree)=subresult(:,2);

    subresult2=checkTreeLeaf(simpRF2{nTree});
    subresult2=sortrows(subresult2,1); % Sort
    NLTCLabelS2(:,nTree)=subresult2(:,2);
end
NLTCLabelS = [NLTCLabelS1 NLTCLabelS2];
% whether noise
NLTCLabelS(NLTCLabelS<NI_threshold)=0; % no
NLTCLabelS(NLTCLabelS>=NI_threshold)=1; % yes


%%
% For each sub from one 0 or 1, 101 for class 1, 101 trees for class  -> binary vote
isNoiseData(1:numTrainSample)=0;  % 1-noise 0-non-noise
for i=1:numTrainSample
    if sum(NLTCLabelS(i,:))>0.5*ntree*nt % voting
        isNoiseData(i)=1;  % nois
    end
end
nonNoiseID = find(isNoiseData==0);
deNoiseData=traindata(nonNoiseID,:); % data after de-noise

end

%% Check the tree and return the the label sequences for all leaf nodes.
% Input:  tree to be detected
% Output: Check the result of the label sequence for all leaves contained in the tree
function [result]=checkTreeLeaf(srTree)
if length(fieldnames(srTree))>2
    resultL=checkTreeLeaf(srTree.leftLeaf);
    resultR=checkTreeLeaf(srTree.rightLeaf);
    result=[resultL;resultR];
else
    datas=srTree.value.datas;
    subresult=checkLabelSequence(srTree.value.labels);
    result=zeros(length(datas),2);
    result(:,1)=datas;
    result(:,2)=subresult;
end
end

%% Check the label Sequence
% Input: label Sequence
% Output: number of level of the last change
function [result]=checkLabelSequence(labels)
nlen=length(labels);
ch1=0;
for i=1:nlen-1
    if labels(i)~=labels(i+1)
        ch1=i; % location of the first change
        break
    end
end
if ch1==0 % Same label sequence 
    result=0;
    return
end
ch2=0;
for i=ch1+1:nlen-1
    if labels(i)~=labels(i+1)
        ch2=i;
        break
    end
end
if ch2==0 % only change once
    ch2=nlen;
end
result=ch2-ch1;
end

% Add sample ID
function [reData]=addOrdNum(data)
[m,~]=size(data);
m=1:m;
m=m';
reData=[data,m];
end

% whether continuous features
% isContiData: 1-continuous features��0-discrete features
function [isContiData]=isContiFunc(data)
[m,n]=size(data);
m=m/3;
isContiData(1,n)=0;
for i=1:n
    % Continuous features: more than one-third of the feature values are different
    if length(unique(data(:,i)))>m
        isContiData(i)=1;
    end
end
end

%% Construct CRDT
% randomly select partition 
% Input data:[label feature ID]
%     isContinuData: whether continuous features,1-continuous,0-discrete
%     upLabels  Label of the parent node
% Output tree: CRDT
%           Non-leaf node: Saves labels for the current and previous nodes
%                       value:labels
%           leaf node: Saves sample ID and labels Sequence
function [tree]=simpRadTree(data,isContinuData,upLabels,flag)
if isempty(data)
    error('The data is empty ...');
end

[m,n]=size(data);

%% check whether leaf node
% leaf node: sample ID and labels Sequence
% non-leaf node: labels
% leaf node evaluation criteria-1: length==1
if m<2
    tree=struct('value','');
    value=struct('datas','','labels','');
    value.datas=data(1,n);
    value.labels=[data(1,1),upLabels];
    tree.value=value;
    return
end
% leaf node evaluation criteria-2: features of the samples in the node are consistent
if length(unique(data(:,1)))==1
    tree=struct('value','');
    value=struct('datas','','labels','');
    value.datas=data(:,n);
    value.labels=[data(1,1),upLabels];
    tree.value=value;
    return
end

%% Partition of non-leaf nodes
% Features and values with random partition 
nNum=n-2;
randArry=randperm(nNum);
flagtemp=0;
for i=1:nNum
    bestAttribute=randArry(i);
    if length(unique(data(:,bestAttribute+1)))>1
        flagtemp=1; % find partition
        break
    end
end

%% data exception occurred
if flagtemp==0
    % feature values are the same but the labels are different
    disp('Wrong ...');
    if length(data(:,1))>5
        data(1:3,1:5)
    else
        data(:,1:5)
    end
    labe=unique(data(:,1));

    if length(labe)==2 
        tree=struct('value','','leftleaf','','rightleaf','');
        value=struct('datas','','labels','');
        value.datas=data(:,n);
        value.labels=upLabels; 
        tree.value=value;

        leftvalue=struct('datas','','labels','');
        leftvalue.datas=data(data(:,1)==labe(1),n);
        leftvalue.labels=[labe(1),upLabels]; % left subtree
        rightvalue=struct('datas','','labels','');
        rightvalue.datas=data(data(:,1)==labe(2),n);
        rightvalue.labels=[labe(2),upLabels]; % right subtree

        leftleaf=struct('value','');
        leftleaf.value=leftvalue;
        rightleaf=struct('value','');
        rightleaf.value=rightvalue;
        tree.leftLeaf=leftleaf;
        tree.rightLeaf=rightleaf;
    else  % The multi-class tree building process is completed by the root node
        disp('multi-class ...');
        labe
        tree=struct('value','');
        value=struct('datas','','labels','');
        value.datas=data(:,n);
        value.labels=[mode(data(:,1)),upLabels];
        tree.value=value;
    end
    return
end

%% Non-leaf nodes divide data in the following manner
randArry=unique(data(:,bestAttribute+1));
bestValueNum=randperm(length(randArry));
if isContinuData(bestAttribute)==1  % continuous feature
    % Select the appropriate partition points according to Otsu
    [~,bestValue] = Otsu4Thres(data(:,bestAttribute+1));

    % partition
    leftData=data(data(:,bestAttribute+1)<=bestValue,:);
    rightData=data(data(:,bestAttribute+1)>bestValue,:);
elseif isContinuData(bestAttribute)==0 % discrete feature
    bestValue=randArry(bestValueNum(1));
    % partition
    leftData=data(data(:,bestAttribute+1)==bestValue,:);
    rightData=data(data(:,bestAttribute+1)~=bestValue,:);
end

tree=struct('value','','leftleaf','','rightleaf','');
value=struct('datas','','labels','');
value.datas=data(:,n);
NodLab = NodeLabel(data,flag,upLabels);%
value.labels=[NodLab,upLabels];
tree.value=value;
tree.leftLeaf=simpRadTree(leftData,isContinuData,[NodLab,upLabels],flag);
tree.rightLeaf=simpRadTree(rightData,isContinuData,[NodLab,upLabels],flag);

end

function [label] = NodeLabel(data,flag,upLabels)
LabelArrary = data(:,1);
SortNum = sortrows(tabulate(LabelArrary),-2);
[RankTop] = SortNum(ParallelSort(SortNum(:,2)));
RankTopNum = length(SortNum(RankTop,1));
if RankTopNum == 1
    label = RankTop(1);
elseif RankTopNum == 2 && flag<=2
    label = RankTop(flag);
elseif RankTopNum == 3
    label = RankTop(flag);
elseif RankTopNum == 2 && flag==3
    group1 = data(find(data(:,1) == RankTop(1)),2:size(data,2)-1);
    group2 = data(find(data(:,1) == RankTop(2)),2:size(data,2)-1);

    [~,ind] = min([mean(pdist(group1)),mean(pdist(group2))]);
    label = RankTop(ind);
    if size(group1,1)==1 && ismember(upLabels(1),RankTop)
        label = upLabels(1);
    end
end

end

%% Otsu: Find the best segmentation threshold in the array
% INPUT: matrix, rows represent samples; columns represent features
% OUTPUT: partition feature and the partition threshold point, Obtaining the maximum variance after the partition
% https://blog.csdn.net/xiaoyucyt/article/details/105956444
function [FeaID,thres] = Otsu4Thres(arrary)
% Ind = randperm(size(matrix,2));
% Sample = data(:,Ind(1:FeaNum));
FeaValueNum = size(arrary,1);
fun = zeros(FeaValueNum,1);
for v = 1:FeaValueNum
    te = arrary(v);
    Group1 = arrary(arrary<=te);
    Group2 = arrary(arrary>te);
    W1 = length(Group1);
    U1 = mean(Group1);
    W2 = length(Group2);
    U2 = mean(Group2);
    U= W1* U1 + W2* U2; % overrall weighted mean
    
    % This measures how far apart the two class means are, weighted by class sizes.
    % A larger value means a better separation if you split at te.
    fun(v)= W1*(U1-U)^2+ W2*( U2-U)^2; % computer betweem-class variance
end

[~,FeaID] = max(fun);
thres = arrary(FeaID);
end


%% Sorting
% INPUT: Vector to be Sorted
% OUTPUT: ID with the most frequency
function [output] = ParallelSort(Frequency)
Frequency = reshape(Frequency,1,length(Frequency));
[y,k] = sort(Frequency,'descend');
idx = [true diff(y)~=0];
ix = k(idx); % unique index from frequency 
t(k) = cumsum(idx); % t is an array with index of k taking values of prefix sum
k(k) = 1:numel(Frequency); %k(k) is the index of the original array
k = k(ix(t));
[output] = find(k==1);
end