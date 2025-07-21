% testing.m

%% 0) locate project root
scriptFile = which('testing.m');
rootDir    = fileparts(scriptFile);

%% 2) make sure MATLAB sees your pipeline code
addpath( genpath(fullfile(rootDir,'matlab')) );

%% 3) override LAMP’s I/O paths
LoadDataPath = fullfile(rootDir,'data', filesep);
SavePath     = fullfile(rootDir,'result', filesep);
if ~exist(SavePath,'dir'), mkdir(SavePath); end
datasetList = {'FBIRN','COBRE'};
%% 4) set up your two-dataset run
DataName     = datasetList;   % {'FBIRN','COBRE'}
SamplingThs  = 0.7;
iter         = 101;
ntree        = 201;
NI_threshold = 2;
TypThs       = 0.4;

%% 5) run the full pipeline
LAMP;   % internally does CountNonNoise → FindTyp → PredictScore

exit;
