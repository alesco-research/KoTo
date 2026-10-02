function [Yout, Zout, Wout] = getMultiStepWindows(Z, W, options)
    %GETMULTISTEPWINDOWS Select representative multi-step windows from trajectories.
    %   [Y, Z, W1, ..., WN] = getMultiStepWindows(Z, W1, ..., WN)
    %   [___] = getMultiStepWindows(___, horizon=H, maxNumberOfWindows=K, ...
    %                               maxNumberOfRawWindows=R)
    %
    %   Extracts all windows of HORIZON steps, clusters them with k-means on
    %   the stacked [start; mid; end] states, keeps the window closest to
    %   each cluster center, and returns the one-step pairs of those windows.
    %
    %   Inputs:
    %     Z   Cell vector, one entry per trajectory. Z{i} is n-by-(T_i+1).
    %     Wj  (Repeating, zero or more.) Cell vector like Z. Wj is
    %         mj-by-T_i.
    %
    %   Name-Value Arguments:
    %     horizon                Window length in steps (default: 20). Must
    %                            be smaller than T_i for at least one i.
    %     maxNumberOfWindows     Max. number of windows, equal to the number
    %                            of k-means clusters (default: 800).
    %     maxNumberOfRawWindows  Random subsample of candidate windows taken
    %                            before clustering (default: no limit).
    %
    %   Outputs:
    %     Yout  n-by-(K*horizon) successor states.
    %     Zout  n-by-(K*horizon) current states.
    %     Wout  (Repeating, one per Wj.) mj-by-(K*horizon) inputs.
    %
    %     K <= maxNumberOfWindows (duplicates are removed). Yout(:,t) is the
    %     successor of Zout(:,t) under Wout{j}(:,t) Rows are ordered
    %     window by window, each in chronological order.
    %
    %   Note: Random result; set rng for reproducibility. Requires the
    %   Statistics and Machine Learning Toolbox.
    %
    %   See also getOneStepPairs, kmeans, dsearchn.
    arguments (Input)
        Z (:,:) cell {mustBeVector};
    end
    arguments (Input, Repeating)
        W (:,:) cell {mustBeVector, mustBeEqualSize(Z, W)};
    end
    arguments (Input)
        options.horizon (1,1) {mustBeInteger, mustBePositive} = 20;
        options.maxNumberOfWindows (1,1) {mustBeInteger, mustBePositive} = 800;
        options.maxNumberOfRawWindows (1,1) {mustBeInteger, mustBePositive} = uint64(inf);
    end
    arguments (Output)
        Yout (:,:)
        Zout (:,:)
    end
    arguments (Output, Repeating)
        Wout (:,:)
    end
    ZStack = [Z{:}];
    WStack = cellfun(@(w) [w{:}], W, UniformOutput=false);
    steps = cellfun(@(z) size(z, 2) - 1, Z(:)');

    % Define index mapping
    inputOffsets = [0, cumsum(steps)];
    stateOffsets = inputOffsets + (0:numel(Z));
    windowOffsets = [0 cumsum(max(0, steps - options.horizon))]; % + 1
    if windowOffsets(end) == 0
        error( ...
            "The provided horizon %d is longer than the " + ...
            "longest trajectory %d.", ...
            options.horizon, max(steps) ...
        )
    end
    
    stateIdxs = NaN(windowOffsets(end), 1);
    inputIdxs = NaN(windowOffsets(end), 1);
    for i = 1:numel(Z)
        wIdx = 1 : steps(i) - options.horizon; % + 1
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
    selectedWindows = (1:numel(stateIdxs));
    if options.maxNumberOfWindows < numel(stateIdxs)
        startMidEnd = [ ...
            ZStack(:, stateIdxs); ... % start
            ZStack(:, stateIdxs + round(options.horizon / 2)); ... % mid
            ZStack(:, stateIdxs + options.horizon) ... % end
        ];
        [~, centers] = kmeans( ...
            startMidEnd', options.maxNumberOfWindows, MaxIter=200, Replicates=3 ...
        );
        selectedWindows = unique(dsearchn(startMidEnd', centers))';
    end

    % Return selected pairs
    k = (0:options.horizon - 1);
    zIdxs = stateIdxs(selectedWindows) + k; % numClusters x horizon
    wIdxs = inputIdxs(selectedWindows) + k; % numClusters x horizon
   
    Yout = ZStack(:, zIdxs(:) + 1);
    Zout = ZStack(:, zIdxs(:));
    Wout = cellfun(@(ws) ws(:, wIdxs(:)), WStack, UniformOutput=false);
end

function mustBeEqualSize(Z, W)
    if numel(W) ~= numel(Z)
        error("Z and W must have the same number of trajectories.");
    end
    if ~all(cellfun(@(z, w) size(z, 2) - 1 == size(w, 2), Z(:), W(:)))
        error("Z and W must have trajectories of matching length.");
    end
end