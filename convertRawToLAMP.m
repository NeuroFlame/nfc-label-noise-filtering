function convertRawToLAMP(srcDir, destDir)
  if ~exist(srcDir,'dir')
    error('Source folder not found: %s', srcDir);
  end
  if ~exist(destDir,'dir')
    mkdir(destDir);
  end

  mats = dir(fullfile(srcDir,'*.mat'));
  for k = 1:numel(mats)
    fname   = mats(k).name;
    [~, ds] = fileparts(fname);
    R       = load(fullfile(srcDir,fname));

    % check required fields
    if ~all(isfield(R,{'FILE_ID','analysis_SCORE','sFNC'}))
      warning('Skipping %s (missing required fields)', fname);
      continue;
    end

    % 1) find diagnosis column
    idx = find(contains(lower(R.FILE_ID),'diagnosis'),1);
    if isempty(idx)
      warning('No diagnosis column found in %s', fname);
      continue;
    end

    % 2) read raw labels (should be 1 or 2)
    rawLabels = R.analysis_SCORE(:,idx);
    if any(~ismember(rawLabels,[1,2]))
      error('Unexpected codes in %s diagnosis: %s', fname, mat2str(unique(rawLabels)'));
    end
    labels = rawLabels;  % now guaranteed to be 1 or 2

    % 3) flatten the 53×53 connectivity into 1×2809 features
    N     = size(R.sFNC,1);
    feats = reshape(R.sFNC, N, []);  % N×2809

    % 4) assemble [labels | features]
    M = [labels, feats];              % N×(1+2809)

    % 5) save under variable named ds
    varName = matlab.lang.makeValidName(ds);
    eval([varName ' = M;']);          %#ok<EVLMT>
    outFile = fullfile(destDir, [varName,'.mat']);
    save(outFile, varName, '-v7.3');
    fprintf('Wrote %s with size %d×%d\n', outFile, size(M));
  end
end
