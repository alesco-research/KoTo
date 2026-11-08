function [Yout, Zout, Wout] = getSingleStepPairs(Z, W, options)
    %GETSINGLESTEPPAIRS Collect single-step transition pairs from trajectories.
    %   [Y, Z, W1, ..., WN] = getSingleStepPairs(Z, W1, ..., WN)
    %   [___] = getSingleStepPairs(___, maxNumberOfPairs=K)
    %
    %   Concatenates all single-step transitions and, if there are more than
    %   K, keeps K of them chosen at random.
    %
    %   Inputs:
    %     Z   Cell vector, one entry per trajectory. Z{i} is n-by-(T_i+1).
    %     Wk  (Repeating, zero or more.) Cell vector like Z. Wk{i} is
    %         mk-by-T_i.
    %
    %   Name-Value Arguments:
    %     maxNumberOfPairs  Max. number of pairs (default: 800).
    %
    %   Outputs:
    %     Yout  n-by-P successor states.
    %     Zout  n-by-P current states.
    %     Wout  (Repeating, one per Wk.) mk-by-P inputs.
    %
    %     P = min(sum(T_i), maxNumberOfPairs). Yout(:,j) is the successor
    %     of Zout(:,j) under Wout{k}(:,j). Columns keep their original
    %     (trajectory by trajectory, chronological) order.
    %
    %   Errors:
    %     Z and Wk differ in trajectory count or lengths.
    %
    %   Note: Random result if subsampling applies; set rng for
    %   reproducibility.
    %
    %   See also getMultiStepWindows.
    arguments (Input)
        Z (:,:) cell {mustBeVector};
    end
    arguments (Input, Repeating)
        W (:,:) cell {mustBeVector, mustBeEqualSize(Z, W)};
    end
    arguments (Input)
        options.maxNumberOfPairs (1,1) {mustBeInteger, mustBePositive} = 800;
    end
    arguments (Output)
        Yout (:,:)
        Zout (:,:)
    end
    arguments (Output, Repeating)
        Wout (:,:)
    end
    Yout = cell2mat(cellfun(@(z) z(:, 2:end), Z(:)', UniformOutput=false));
    Zout = cell2mat(cellfun(@(z) z(:, 1:end-1), Z(:)', UniformOutput=false));
    Wout = cellfun(@(w) [w{:}], W, UniformOutput=false);

    % Limit number of pairs
    numberOfPairs = size(Zout, 2);
    if size(Zout, 2) > options.maxNumberOfPairs
        keep = sort(randperm(numberOfPairs, options.maxNumberOfPairs));
        Yout = Yout(:, keep);
        Zout = Zout(:, keep);
        Wout = cellfun(@(w) w(:, keep), Wout, UniformOutput=false);
    end
end

function mustBeEqualSize(Z, W)
    if numel(W) ~= numel(Z)
        error("Z and W must have the same number of trajectories.");
    end
    if ~all(cellfun(@(z, w) size(z, 2) - 1 == size(w, 2), Z(:), W(:)))
        error("Z and W must have trajectories of matching length.");
    end
end