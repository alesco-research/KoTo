classdef IODelayEmbedding
    %IODELAYEMBEDDING Input-output delay embedding for data-driven predictors.
    %   obj = IODelayEmbedding(outputDim, inputDim, numberOfOutputs, numberOfInputs)
    %   builds the state from the last numberOfOutputs measured outputs and
    %   the last numberOfInputs applied inputs,
    %
    %       q_t = [y_t; y_{t-1}; ...; y_{t-ny+1}; u_{t-1}; ...; u_{t-nu}],
    %
    %   with ny = numberOfOutputs and nu = numberOfInputs. The input applied
    %   at time t, u_t, is NOT part of the state; it is the input of the
    %   one-step map q_{t+1} = F(q_t, u_t). For a linear system of order n
    %   with ny, nu >= n (observability index) q_t is a state.
    %
    %   [Q, W] = embed(obj, Y, U)
    %       Y is outputDim-by-N (columns are samples), U is inputDim-by-T
    %       with N = T or N = T+1 (the usual case: the output trajectory
    %       includes the sample after the last input). Q contains q_t for
    %       t = memory+1, ..., N and W contains u_t for t = memory+1, ..., T,
    %       so Q(:, k+1) is the successor of Q(:, k) under W(:, k).
    %       Embed each experiment separately, never across experiments.
    %
    %   Y = unembed(obj, Q)
    %       Return the current output block Q(1:outputDim, :).
    %
    %   Properties (read-only):
    %      outputDim, inputDim             - Signal dimensions
    %      numberOfOutputs, numberOfInputs - Past samples in the state
    %      memory      - First usable sample index minus one, max(ny-1, nu)
    %      featureDim  - Dimension of q, ny*outputDim + nu*inputDim

    properties (SetAccess=immutable)
        outputDim (1,1) {mustBeInteger, mustBePositive} = 1;
        inputDim (1,1) {mustBeInteger, mustBePositive} = 1;
        numberOfOutputs (1,1) {mustBeInteger, mustBePositive} = 1;
        numberOfInputs (1,1) {mustBeInteger, mustBeNonnegative} = 0;
        memory (1,1) {mustBeInteger, mustBeNonnegative} = 0;
        featureDim (1,1) {mustBeInteger, mustBePositive} = 1;
    end

    methods
        function obj = IODelayEmbedding(outputDim, inputDim, numberOfOutputs, numberOfInputs)
            arguments
                outputDim (1,1) {mustBeInteger, mustBePositive};
                inputDim (1,1) {mustBeInteger, mustBePositive};
                numberOfOutputs (1,1) {mustBeInteger, mustBePositive};
                numberOfInputs (1,1) {mustBeInteger, mustBeNonnegative};
            end
            obj.outputDim = outputDim;
            obj.inputDim = inputDim;
            obj.numberOfOutputs = numberOfOutputs;
            obj.numberOfInputs = numberOfInputs;
            obj.memory = max(numberOfOutputs - 1, numberOfInputs);
            obj.featureDim = numberOfOutputs * outputDim + numberOfInputs * inputDim;
        end

        function [Q, W] = embed(obj, Y, U)
            arguments
                obj (1,1) IODelayEmbedding;
                Y (:,:) {mustBeNumeric};
                U (:,:) {mustBeNumeric, IODelayEmbedding.mustBeEmbeddable(obj, Y, U)};
            end
            [ny, nu] = deal(obj.numberOfOutputs, obj.numberOfInputs);
            [p, m] = deal(obj.outputDim, obj.inputDim);

            times = obj.memory + 1:size(Y, 2);
            Q = zeros(obj.featureDim, numel(times));
            for i = 0:ny - 1
                Q(i * p + (1:p), :) = Y(:, times - i);
            end
            for i = 1:nu
                Q(ny * p + (i - 1) * m + (1:m), :) = U(:, times - i);
            end
            W = U(:, obj.memory + 1:end);
        end

        function Y = unembed(obj, Q)
            arguments
                obj (1,1) IODelayEmbedding;
                Q (:,:) {mustBeNumeric, IODelayEmbedding.mustBeEmbedding(obj, Q)};
            end
            Y = Q(1:obj.outputDim, :);
        end
    end

    methods (Access=private, Hidden, Static)
        function mustBeEmbeddable(obj, Y, U)
            if size(Y, 1) ~= obj.outputDim || size(U, 1) ~= obj.inputDim
                error( ...
                    "IODelayEmbedding:DimensionMismatch", ...
                    "Expected %d output and %d input row(s), got %d and %d.", ...
                    obj.outputDim, obj.inputDim, size(Y, 1), size(U, 1));
            end
            if size(Y, 2) ~= size(U, 2) && size(Y, 2) ~= size(U, 2) + 1
                error( ...
                    "IODelayEmbedding:LengthMismatch", ...
                    "Outputs must have as many samples as inputs, or one more.");
            end
            if size(Y, 2) < obj.memory + 1
                error( ...
                    "IODelayEmbedding:TooFewData", ...
                    "Trajectory must have at least memory+1 = %d samples.", ...
                    obj.memory + 1);
            end
        end

        function mustBeEmbedding(obj, Q)
            if size(Q, 1) ~= obj.featureDim
                error( ...
                    "IODelayEmbedding:DimensionMismatch", ...
                    "Embedding must have featureDim = %d rows, got %d.", ...
                    obj.featureDim, size(Q, 1));
            end
        end
    end
end
