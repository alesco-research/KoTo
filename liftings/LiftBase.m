classdef (Abstract) LiftBase < handle
    %LIFTBASE Common interface for all feature liftings.
    %   Every concrete lifting (PolynomialLift, ChebyshevLift, KernelLift
    %   subclasses, CompositeLift, ...) inherits from this class. It
    %   defines the lifting interface and adds a learned linear
    %   projection that maps lifted features back to the original state.
    %
    %   LiftBase is a handle class: fitProjection modifies the object in
    %   place, and copies of a lifting object share the same projection.
    %
    %   Subclasses must provide (abstract properties, SetAccess=protected):
    %     - dim          : dimension of the original state x (rows of X)
    %     - featureDim   : dimension of the lifted feature space (rows of Z)
    %     - includeState : true if the lifting contains the state in the
    %                      layout Z = [1; x; ...], i.e. a constant one,
    %                      then the state, then any further features
    %   and the abstract method:
    %     - lift(obj, X) : nonlinear transform that maps data 
    %                      X (dim x nSamples) to Z (featureDim x nSamples)
    %
    %   Projection (lifted space -> state):
    %     The property "projection" is a dim-by-featureDim matrix P such
    %     that x is approximately P*z. It is empty until fitProjection
    %     has been called.
    %
    %     err = fitProjection(obj, X)
    %     err = fitProjection(obj, X, gamma=g, XTest=Xt)
    %         Fits P on training data X and overwrites any existing
    %         projection. Returns the RMSE of P*lift(XTest) vs. XTest;
    %         XTest defaults to X (in-sample error).
    %         - includeState = true: P = [0, I, 0]. The layout
    %           Z(1:dim+1,:) == [1; X] is checked exactly, else
    %           "Lift:StateLayout" is raised. gamma is ignored, error is 0.
    %         - includeState = false: ridge regression via normal
    %           equations, P = X*Z' / (Z*Z' + gamma*I). gamma = 0 is
    %           plain least squares and can be singular or
    %           ill-conditioned; use gamma > 0 in that case. A
    %           non-finite P is stored anyway and rejected later by
    %           getReprojectionError and project.
    %
    %     err = getReprojectionError(obj, X)
    %         RMSE over all entries between P*lift(X) and X for a fitted
    %         projection.
    %
    %     XP = project(obj, Z)
    %         Maps lifted data Z (featureDim x nSamples) back to the
    %         state space, XP = P*Z (dim x nSamples).
    %
    %   Static validators for use in subclasses' arguments blocks:
    %     - mustMatchDim(obj, val)         : size(val,1) == obj.dim
    %     - mustMatchFeatureDim(obj, val)  : size(val,1) == obj.featureDim
    %     - mustHaveValidProjection(obj)   : projection is fitted and finite
    %
    %   Error identifiers: Lift:DimensionMismatch, Lift:StateLayout,
    %   Lift:ProjectionUndefined, Lift:NonFiniteProjection.

    properties (Abstract, SetAccess=protected)
        includeState (1,1) logical
        dim (1,1) {mustBeInteger, mustBePositive}
        featureDim (1,1) {mustBeInteger, mustBePositive}
    end

    properties (SetAccess=protected)
        projection (:,:) {mustBeNumeric} = [];
    end

    methods (Abstract)
        Z = lift(obj, X)
    end

    methods
        function reprojectionError = fitProjection(obj, X, options)
            arguments
                obj (1,1) LiftBase;
                X (:,:) {mustBeNumeric, LiftBase.mustMatchDim(obj, X)};
                options.gamma (1,1) {mustBeNumeric, mustBeNonnegative} = 0;
                options.XTest (:,:) {mustBeNumeric, LiftBase.mustMatchDim(obj, options.XTest)} = X;
            end
            Z = obj.lift(X);
            if obj.includeState
                if ~isequal(Z(1:obj.dim + 1, :), [ones(1, size(X, 2)); X])
                    error( ...
                        "Lift:StateLayout", ...
                        "includeState=true requires the layout [1; X; ...]." ...
                    );
                end
                obj.projection = [ ...
                    zeros(obj.dim, 1), ...
                    eye(obj.dim), ...
                    zeros(obj.dim, obj.featureDim - obj.dim - 1) ...
                ];
            else
                obj.projection = ( ...
                    (X * Z') / (Z * Z' + options.gamma * eye(obj.featureDim)) ...
                );
            end
            reprojectionError = obj.getReprojectionError(options.XTest);
        end

        function reprojectionError = getReprojectionError(obj, X)
            arguments
                obj (1,1) LiftBase {LiftBase.mustHaveValidProjection(obj)};
                X (:,:) {mustBeNumeric, LiftBase.mustMatchDim(obj, X)};
            end
            reprojectionError = mean( ...
                sqrt(sum((obj.projection * obj.lift(X) - X).^2, 1)) ...
            );
        end

        function XP = project(obj, Z)
            arguments
                obj (1,1) LiftBase {LiftBase.mustHaveValidProjection(obj)};
                Z (:,:) {mustBeNumeric, LiftBase.mustMatchFeatureDim(obj, Z)};
            end
            XP = obj.projection * Z;
        end
    end

    methods (Static, Hidden, Access=protected)
        function mustMatchDim(obj, val)
            if size(val, 1) ~= obj.dim
                error( ...
                    "Lift:DimensionMismatch", ...
                    "Value must have %d row(s) to match the transformation.", ...
                    obj.dim ...
                );
            end
        end

        function mustMatchFeatureDim(obj, val)
            if size(val, 1) ~= obj.featureDim
                error( ...
                    "Lift:DimensionMismatch", ...
                    "Value must have %d row(s) to match the projection.", ...
                    obj.featureDim ...
                );
            end
        end

        function mustHaveValidProjection(obj)
            if isempty(obj.projection)
                error( ...
                    "Lift:ProjectionUndefined", ...
                    "To project data first fit the projection " + ...
                    "by calling `fitProjection(obj, X)`." ...
                );
            end
            if ~all(isfinite(obj.projection), "all")
                error( ...
                    "Lift:NonFiniteProjection", ...
                    "The projection matrix contains non-finite values " + ...
                    "and is likely singular." ...
                );  
            end
        end
    end
end
