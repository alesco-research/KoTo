classdef Scale
    %SCALE Affine (shift + scaling) map between an input range and an output range.
    %   obj = SCALE(inputRange) maps each dimension from inputRange to [0 1].
    %   inputRange is N-by-2, with rows [lower upper] for each of N dimensions.
    %
    %   obj = SCALE(inputRange, outputRange) maps onto outputRange (N-by-2)
    %   instead, e.g. repmat([-1 1], N, 1).
    %
    %   Y = scale(obj, X)     forward map  y = A*x + b.
    %   X = unscale(obj, Y)   inverse map  x = A^-1*(y - b).
    %
    %   Columns of X and Y are observations; rows are dimensions.
    %
    %   Scale properties (read-only):
    %      inputRange            - Original data range, N-by-2
    %      outputRange           - Scaled data range, N-by-2
    %      transformation        - Scaling matrix A (diagonal)
    %      inverseTransformation - Inverse scaling matrix A^-1
    %      shift                 - Offset vector b, applied after scaling
    %
    %   Scale methods:
    %      scale     - Apply the forward map (x -> y)
    %      unscale   - Apply the inverse map (y -> x)
    %
    %   Errors: Scale:NonFiniteRange, Scale:InvalidRange (lower >= upper),
    %   Scale:DimensionMismatch (row count differs from N).

    properties (GetAccess=public, SetAccess=immutable)
        inputRange (:,2) {mustBeNumeric}
        outputRange (:,2) {mustBeNumeric}
        transformation (:,:) {mustBeNumeric}
        inverseTransformation (:,:) {mustBeNumeric}
        shift (:,1) {mustBeNumeric}
    end

    properties (Hidden, SetAccess=immutable)
        dim (1,1) {mustBeInteger};
    end

    methods
        function obj = Scale(inputRange, outputRange)
            arguments
                inputRange (:,2) {mustBeNumeric, Scale.mustBeValidRange};
                outputRange (:,2) {mustBeNumeric, Scale.mustBeValidRange} = [zeros(size(inputRange, 1), 1), ones(size(inputRange, 1), 1)];
            end
            obj.dim = size(inputRange, 1);
            obj.inputRange = inputRange;
            obj.outputRange = outputRange;
            obj.mustMatchDim(obj, outputRange)
            
            ratio = diff(outputRange, 1, 2) ./ diff(inputRange, 1, 2);
            obj.transformation = diag(ratio);
            obj.inverseTransformation = diag(1 ./ ratio);
            obj.shift = outputRange(:,1) - ratio .* inputRange(:,1);
        end

        function scaledData = scale(obj, data)
            arguments
                obj (1,1) Scale
                data (:,:) {mustBeNumeric, Scale.mustMatchDim(obj, data)}
            end
            scaledData = obj.transformation * data + obj.shift;
        end

        function data = unscale(obj, scaledData)
            arguments
                obj (1,1) Scale
                scaledData (:,:) {mustBeNumeric, Scale.mustMatchDim(obj, scaledData)}
            end
            data = obj.inverseTransformation * (scaledData - obj.shift);
        end
    end

    methods (Access=private, Hidden, Static)
        function mustBeValidRange(val)
            if ~all(isfinite(val), "all")
                error( ...
                    "Scale:NonFiniteRange", ...
                    "Range values must be finite.");
            end
            if any(val(:,2) <= val(:,1))
                error( ...
                    "Scale:InvalidRange", ...
                    "Each range row must satisfy lower < upper.");
            end
        end

        function mustMatchDim(obj, val)
            if size(val,1) ~= obj.dim
                error( ...
                    "Scale:DimensionMismatch", ...
                    "Input value must have %d row(s).", obj.dim);
            end
        end
    end
end