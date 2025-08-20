function convertRawToLAMP_triangle(srcDir, destDir)
% convertRawToLAMP_triangle  Convert raw .mat files into LAMP pipeline format by
% extracting the lower triangle (including diagonal) of sFNC and appending labels
% in the last column.
% 
%   convertRawToLAMP_triangle(srcDir, destDir)
% 
%   - srcDir  : folder containing original .mat files with fields:
%                 FILE_ID (1×F cell), analysis_SCORE (N×F), sFNC (N×P×P)
%   - destDir : folder where processed .mat files will be written (created if needed)
% 
% Each output .mat will contain exactly one variable, named after the cohort
% (e.g. "FBIRN"), of size N×(P*(P+1)/2 + 1):
%   cols 1:end-1 = flattened lower-triangle connectivity (including diagonal)
%   col end       = diagnosis labels (1 or 2)

  if nargin<2
    error('Usage: convertRawToLAMP_triangle(srcDir, destDir)');
  end
  if ~exist(srcDir,'dir')
    error('Source directory does not exist: %s', srcDir);
  end
  if ~exist(destDir,'dir')
    mkdir(destDir);
  end

  files = dir(fullfile(srcDir,'*.mat'));
  for k = 1:numel(files)
    fname = files(k).name;
    [~, dsName] = fileparts(fname);
    srcPath = fullfile(srcDir, fname);

    % Load raw data
    R = load(srcPath);
    required = {'FILE_ID','analysis_SCORE','sFNC'};
    if ~all(isfield(R, required))
      warning('Skipping %s: missing fields', fname);
      continue;
    end

    % Find diagnosis column
    fileIDs = R.FILE_ID(:);
    diagIdx = find(contains(lower(fileIDs),'diagnosis'),1);
    if isempty(diagIdx)
      warning('No diagnosis column found in %s', fname);
      continue;
    end

    % Extract labels (1=SZ, 2=HC)
    rawLabels = R.analysis_SCORE(:, diagIdx);
    if any(~ismember(rawLabels,[1,2]))
      error('Unexpected diagnosis codes in %s: %s', fname, mat2str(unique(rawLabels)'));
    end
    labels = rawLabels;

    % Flatten only lower triangle (including diagonal) of sFNC
    N = size(R.sFNC,1);
    P = size(R.sFNC,2);
    disp(['Processing dataset: ' dsName ', subjects: ' num2str(N) ', features: ' num2str(P)]);
    mask = tril(true(P), -1);            
    idx  = find(mask);
    allFeat = reshape(R.sFNC, N, P*P);
    feats   = allFeat(:, idx);          % N×(P*(P+1)/2)

    % Assemble matrix [features | labels]
    M = [feats, labels];                % N×(P*(P+1)/2 + 1)

    % Save under variable named after dataset
    varName = matlab.lang.makeValidName(dsName);
    eval(sprintf('%s = M;', varName));  %#ok<EVLMT>
    outPath = fullfile(destDir, [varName,'.mat']);
    save(outPath, varName, '-v7.3');
    fprintf('Wrote %s (%d subjects × %d features + label)\n', outPath, N, size(feats,2));
  end
end
