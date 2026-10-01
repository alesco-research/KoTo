classdef ChebyshevLift < LiftBase
    %CHEBYSHEVLIFT Chebyshev polynomial basis lifting (1D inputs only).

    properties (SetAccess=protected)
        dim = 1;
        featureDim = 1;
        includeState = true;
        degree (1,1) {mustBeInteger, mustBePositive} = 1;
    end

    methods
        function obj = ChebyshevLift(degree)
            arguments
                degree (1,1) {mustBeInteger, mustBePositive};
            end
            obj.degree = degree;
            obj.featureDim = nchoosek(obj.dim + degree, degree);
        end

        function Z = lift(obj, X)
            arguments
                obj (1,1) ChebyshevLift;
                X (:,:) {mustBeNumeric, LiftBase.mustMatchDim(obj, X)};
            end
            Z = zeros(obj.degree + 1, numel(X));
            Z(1,:) = 1;
            if obj.degree >= 1, Z(2,:) = X; end
            for k = 2:obj.degree
                Z(k+1,:) = 2 .* X .* Z(k,:) - Z(k-1,:);
            end
        end
    end
end
