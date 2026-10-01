function [Yout, Zout, Wout] = getMultiStepWindows(Z, W, horizon, maxNumberOfWindows, options)
    %GETMULTISTEPWINDOWS Select representative multi-step windows from trajectories.
    %   [Y, Z, W] = getMultiStepWindows(Z, W, horizon, maxNumberOfWindows)
    %   [Y, Z, W] = getMultiStepWindows(___, maxNumberOfRawWindows=N)
    %
    %   Extracts all windows of length HORIZON from the trajectories,
    %   picks representative ones via k-means on the stacked
    %   [start; mid; end] states, and returns their one-step pairs.
    %
    %   Inputs:
    %     Z, W  Cell vectors, one entry per trajectory. Z{i} is n-by-(T_i+1),
    %           W{i} is m-by-T_i.
    %     horizon              Window length in steps. Must be shorter than
    %                          at least one trajectory, else error
    %                          Koopman:HorizonTooLong.
    %     maxNumberOfWindows   Max. number of windows / k-means clusters
    %                          (default: 800).
    %
    %   Name-Value Arguments:
    %     maxNumberOfRawWindows  Random subsample of candidate windows taken
    %                            before clustering (default: no limit).
    %
    %   Outputs:
    %     Yout, Zout, Wout  Matrices with K*horizon columns, K <= maxNumberOfWindows
    %                       (duplicates are removed). Yout(:,j) is the
    %                       successor of Zout(:,j) under Wout(:,j); columns
    %                       are ordered window by window.
    %
    %   Note: Random result. Requires Statistics and Machine Learning Toolbox.
    %
    %   See also getOneStepPairs, kmeans, dsearchn.
    arguments
        Z (:,:) cell {mustBeVector};
        W (:,:) cell {mustBeVector};
        horizon (1,1) {mustBeInteger, mustBePositive};
        maxNumberOfWindows (1,1) {mustBeInteger, mustBePositive} = 800;
        options.maxNumberOfRawWindows (1,1) {mustBeInteger, mustBePositive} = uint64(inf);
    end
    [ZStack, WStack] = deal([Z{:}], [W{:}]);
    steps = cellfun(@(uv) size(uv,2), W);

    % Define index mapping
    inputOffsets = [0, cumsum(steps(:)')];
    stateOffsets = inputOffsets + (0:numel(W));
    windowOffsets = [0 cumsum(max(0, steps(:)' - horizon))];
    if windowOffsets(end) == 0
        error( ...
            "Koopman:HorizonTooLong", ...
            "The provided horizon %d is longer than the " + ...
            "longest trajectory %d.", ...
            horizon, max(steps(:)) ...
        )
    end
    
    stateIdxs = NaN(windowOffsets(end), 1);
    inputIdxs = NaN(windowOffsets(end), 1);
    for i = 1:numel(W)
        wIdx = 1 : steps(i) - horizon;
        stateIdxs(windowOffsets(i) + wIdx) = stateOffsets(i) + wIdx;
        inputIdxs(windowOffsets(i) + wIdx) = inputOffsets(i) + wIdx;
    end
    
    % Limit number of raw windows
    if windowOffsets(end) > options.maxNumberOfRawWindows
        keep = randperm( ...
            windowOffsets(end), options.maxNumberOfRawWindows ...
        );
        stateIdxs = stateIdxs(keep);
        inputIdxs = inputIdxs(keep);
    end

    % Find expressive windows using kmeans
    numClusters = min(maxNumberOfWindows, numel(stateIdxs));
    startMidEnd = [ ...
        ZStack(:, stateIdxs); ... % start
        ZStack(:, stateIdxs + round(horizon / 2)); ... % mid
        ZStack(:, stateIdxs + horizon) ... % end
    ];
    [~, centers] = kmeans( ...
        startMidEnd', numClusters, MaxIter=200, Replicates=3 ...
    );
    selectedWindows = unique(dsearchn(startMidEnd', centers))';

    % Return selected pairs
    k = (0:horizon - 1);
    zIdxs = stateIdxs(selectedWindows) + k; % numClusters x horizon
    wIdxs = inputIdxs(selectedWindows) + k; % numClusters x horizon

    Wout = WStack(:, wIdxs(:));
    Zout = ZStack(:, zIdxs(:));
    Yout = ZStack(:, zIdxs(:) + 1);
end