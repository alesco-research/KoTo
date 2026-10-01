classdef (Abstract) KernelLift < LiftBase
    %KERNELLIFT Common interface for all kernel-based feature liftings
    %(evaluating a fixed kernel at a set of center points). Concrete
    %subclasses (GaussianLift, Matern52Lift, RationalQuadraticLift, ...)
    %only need to implement the protected kernel(obj, x, y) method.

    properties (SetAccess=protected)
        dim = 1;
        featureDim = 1;
        includeState = false;
        centers (:,:) {mustBeNumeric} = [];
    end

    methods (Sealed)
        function obj = KernelLift(dim, centers, options)
            arguments
                dim (1,1) {mustBeInteger, mustBePositive};
                centers (:,:) {mustBeNumeric};
                options.includeState (1,1) logical = false;
            end
            obj.dim = dim;
            LiftBase.mustMatchDim(obj, centers);
            obj.centers = centers;
            obj.includeState = options.includeState;
            obj.featureDim = obj.dim * options.includeState + size(centers, 2);
        end

        function Z = lift(obj, X)
            arguments
                obj (1,1) KernelLift;
                X (:,:) {mustBeNumeric, LiftBase.mustMatchDim(obj, X)};
            end
            nx = size(X, 2);
            nc = size(obj.centers, 2);

            Z = zeros(nc, nx);
            for i = 1:nc
                Z(i,:) = obj.kernel(X, obj.centers(:,i) * ones(1, nx));
            end
            Z = sqrt(2/nc) * Z;
            if obj.includeState
                Z = [ones(1, nx); X; Z];
            end
        end
    end

    methods (Abstract, Access=protected)
        k = kernel(obj, x, y)
    end
end
