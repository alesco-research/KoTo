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
    end

    methods (Abstract)
        y = predict(obj, z, u)
        varargout = getOperatorNorm()
    end
    
    methods (Sealed)
        function Z = rollout(obj, z0, w)
            arguments
                obj (1,1) KoopmanBase {KoopmanBase.mustBeFitted(obj)};
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

    methods (Static, Hidden, Access=protected)
        function P = columnKron(A, B)
            % P(:,k) = kron(A(:,k), B(:,k)) for every column k
            n = size(A, 2);
            P = reshape(reshape(B, size(B,1), 1, n) .* reshape(A, 1, size(A,1), n), [], n);
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
