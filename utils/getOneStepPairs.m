function [Yout, Zout, Wout] = getOneStepPairs(Z, W, maxNumberOfPairs)
    %GETONESTEPPAIRS Collect random one-step transition pairs from trajectories.
    %   [Y, Z, W] = getOneStepPairs(Z, W, maxNumberOfPairs)
    %
    %   Concatenates all one-step transitions and keeps at most
    %   MAXNUMBEROFPAIRS of them, chosen at random.
    %
    %   Inputs:
    %     Z, W  Cell vectors, one entry per trajectory. Z{i} is n-by-(T_i+1),
    %           W{i} is m-by-T_i.
    %     maxNumberOfPairs  Max. number of pairs (default: 800).
    %
    %   Outputs:
    %     Yout, Zout, Wout  Matrices with min(sum(T_i), maxNumberOfPairs)
    %                       columns in random order. Yout(:,j) is the
    %                       successor of Zout(:,j) under Wout(:,j).
    %
    %   See also getMultiStepWindows.
    arguments
        Z (:,:) cell {mustBeVector};
        W (:,:) cell {mustBeVector};
        maxNumberOfPairs (1,1) {mustBeInteger, mustBePositive} = 800;
    end
    Yout = cell2mat(cellfun(@(z) z(:, 2:end), Z(:), UniformOutput=false)');
    Zout = cell2mat(cellfun(@(z) z(:, 1:end-1), Z(:), UniformOutput=false)');
    Wout = [W{:}];
    
    % Limit number of pairs
    numberOfPairs = min(size(Wout, 2), maxNumberOfPairs);
    keep = randperm(size(Wout, 2), numberOfPairs);
    Yout = Yout(:, keep);
    Zout = Zout(:, keep);
    Wout = Wout(:, keep);
end