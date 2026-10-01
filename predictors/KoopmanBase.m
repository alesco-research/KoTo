classdef (Abstract) KoopmanBase < handle
    %KOOPMANBASE Abstract base class for Koopman models (handle class).
    %   Subclasses implement predict and learn_. The sealed learn method
    %   builds training pairs from trajectories and passes them to learn_.
    %
    %   obj = Subclass(Name=Value, ...)
    %     lifted features.
    %     useMultiStepWindows (true)  Pair selection mode:
    %       true : requires horizon, maxNumberOfWindows;
    %              optional maxNumberOfRawWindows. Expressiv windows are 
    %              chosen by k-means on [start; mid; end] states.
    %       false: requires maxNumberOfPairs (random one-step pairs).
    %
    %   [Y, Z, W] = learn(obj, Zc, Wc, varargin)
    %     Zc, Wc: cell vectors, one entry per trajectory; Zc{i} has one
    %     more column than Wc{i}. Returns successor states Y, start
    %     states Z and inputs W with matching columns, and passes them
    %     plus varargin to learn_.
    %
    %   Static: getMultiStepWindows(Z, W, horizon, ...),
    %           getOneStepPairs(Z, W, ...)

    properties (SetAccess=public)
        stateDim (1,1) {mustBeInteger} = 0;
        inputDim (1,1) {mustBeInteger} = 0;
        getPairs (1,1) function_handle = @empty;
    end

    methods (Abstract)
        y = predict(obj, z, u)
        varargout = getOperatorNorm()
    end

    methods (Abstract, Hidden, Access=protected)
        learn_(obj, Y, Z, W, varargin)
    end
    
    methods (Sealed)
        function obj = KoopmanBase(pairOptions, options)
            arguments
                pairOptions.horizon (1,1) {mustBeInteger, mustBePositive};
                pairOptions.maxNumberOfWindows (1,1) {mustBeInteger, mustBePositive};
                pairOptions.maxNumberOfRawWindows (1,1) {mustBeInteger, mustBePositive};
                pairOptions.maxNumberOfPairs (1,1) {mustBeInteger, mustBePositive};
                options.useMultiStepWindows (1,1) logical = true;
            end
            if options.useMultiStepWindows
                positional = ["horizon", "maxNumberOfWindows"];
                optional = "maxNumberOfRawWindows";
                args = obj.getArguments(pairOptions, positional, optional);
                obj.getPairs = @(Z, W) obj.getMultiStepWindows(Z, W, args{:});
            else
                positional = "maxNumberOfPairs";
                args = obj.getArguments(pairOptions, positional, []);
                obj.getPairs = @(Z, W) obj.getOneStepPairs(Z, W, args{:});
            end            
        end

        function [Yout, Zout, Wout] = learn(obj, Z, W, varargin)
            arguments
                obj (1,1) KoopmanBase;
                Z (:,:) cell {mustBeVector};
                W (:,:) cell {mustBeVector};
            end
            arguments (Repeating)
                varargin
            end
            obj.stateDim = size(Z{1}, 1);
            obj.inputDim = size(W{1}, 1);

            [Yout, Zout, Wout] = obj.getPairs(Z, W);
            obj.learn_(Yout, Zout, Wout, varargin{:});
        end

        function Z = rollout(obj, z0, w)
            arguments
                obj (1,1) LinearKoopman {KoopmanBase.mustBeFitted(obj)};
                z0 (:,:) {KoopmanBase.mustMatchStateDim(obj, z0)};
                w (:,:) {KoopmanBase.mustMatchInputDim(obj, w)};
            end
            T = size(w, 2);
            Z = zeros(obj.stateDim, T+1); Z(:, 1) = z0(:);
            for t = 1:T
                Z(:, t+1) = obj.predict(Z(:,t), w(:,t));
            end
        end
    end

    methods (Static)
        function [Yout, Zout, Wout] = getMultiStepWindows(Z, W, horizon, maxNumberOfWindows, options)
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

        function [Yout, Zout, Wout] = getOneStepPairs(Z, W, maxNumberOfPairs)
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
    end

    methods (Static, Hidden, Access=protected)
        function P = columnKron(A, B)
            % P(:,k) = kron(A(:,k), B(:,k)) for every column k
            n = size(A, 2);
            P = reshape(reshape(B, size(B,1), 1, n) .* reshape(A, 1, size(A,1), n), [], n);
        end

        function args = getArguments(options, positional, optional)
            providedArguments = fieldnames(options);
            unexpectedArguments = providedArguments( ...
                ~ismember(providedArguments, [positional, optional]) ...
            );
            if ~isempty(unexpectedArguments)
                error( ...
                    "Koopman:UnexpectedArgument", ...
                    "Unexpected argument(s): %s.", ...
                    strjoin(unexpectedArguments, ", ") ...
                )
            end
            missingArguments = positional( ...
                ~ismember(positional, providedArguments) ...
            );
            if ~isempty(missingArguments)
                error( ...
                    "Koopman:MissingArgument", ...
                    "Missing required argument(s): %s.", ...
                    strjoin(missingArguments, ", ") ...
                )
            end

            args = cell(1, numel(positional));
            for i = 1:numel(positional)
                args{i} = options.(positional(i));
            end
            args = [args, namedargs2cell(rmfield(options, positional))];
        end

        function mustBeFitted(obj)
            if obj.inputDim == 0 || obj.stateDim == 0
                error( ...
                    "Koopman:OperatorUndefined", ...
                    "To use `predict` first fit the Koopman operator " + ...
                    "using `learn`." ...
                );
            end
        end
        
        function mustMatchStateDim(obj, val)
            if size(val, 1) ~= obj.stateDim
                error( ...
                    "Koopman:DimensionMismatch", ...
                    "Value must have %d row(s) to match the " + ... 
                    "Koopman operator.", ...
                    obj.stateDim ...
                );
            end
        end

        function mustMatchInputDim(obj, val)
            if size(val, 1) ~= obj.inputDim
                error( ...
                    "Koopman:DimensionMismatch", ...
                    "Value must have %d row(s) to match the " + ... 
                    "Koopman operator.", ...
                    obj.inputDim ...
                );
            end
        end
    end
end
