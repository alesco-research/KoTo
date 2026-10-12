classdef (Abstract) KoopmanBase < handle
    %KoopmanBase Abstract base class for dynamical-system predictors (handle class).
    %   Subclasses implement predict, which maps a state and an input to the
    %   successor state. The sealed rollout method uses predict to simulate
    %   a trajectory over an input sequence.
    %
    %   Properties (public):
    %     stateDim  Number of state rows (positive integer scalar).
    %     inputDim  Number of input rows (positive integer scalar).
    %
    %   y = predict(obj, z, u)    (abstract)
    %     z: state column vector (stateDim x 1), u: input column vector
    %     (inputDim x 1). Returns the predicted successor state y.
    %
    %   Z = rollout(obj, z0, w)   (sealed)
    %     z0: initial state (stateDim x 1), w: input sequence
    %     (inputDim x T). Returns the trajectory Z (stateDim x T+1),
    %     where Z(:,1) = z0 and Z(:,t+1) = predict(obj, Z(:,t), w(:,t)).
    %
    %   Protected static helpers for subclasses:
    %     columnKron(A, B)           Column-wise Kronecker product.
    %     mustBeEqualSize(Z, W)      Check equal number of columns.
    %     mustMatchStateDim(obj, v)  Check size(v,1) == obj.stateDim.
    %     mustMatchInputDim(obj, v)  Check size(v,1) == obj.inputDim.

    properties (SetAccess=public)
        stateDim {mustBeInteger, mustBePositive} = [];
        inputDim {mustBeInteger, mustBePositive} = [];
    end

    methods (Abstract)
        y = predict(obj, z, u)
    end
    
    methods (Sealed)
        function Z = rollout(obj, z0, w)
            arguments
                obj (1,1) KoopmanBase;
                z0 (:,1) {KoopmanBase.mustMatchStateDim(obj, z0)};
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
            n = size(A, 2);
            P = reshape(reshape(B, size(B,1), 1, n) .* reshape(A, 1, size(A,1), n), [], n);
        end

        function mustBeEqualSize(Z, W)
            if size(Z, 2) ~= size(W, 2)
                error( ...
                    "KoopmanBase:DimensionMismatch", ...
                    "Inputs must have the same number of samples. " + ...
                    "Received inputs of dimension %d and %d instead.", ...
                    size(Z, 2), ...
                    size(W, 2) ...
            );
            end
        end
        
        function mustMatchStateDim(obj, val)
            if size(val, 1) ~= obj.stateDim
                error( ...
                    "KoopmanBase:DimensionMismatch", ...
                    "Value must have %d row(s) to match the " + ... 
                    "state dimension.", ...
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
