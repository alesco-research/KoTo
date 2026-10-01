classdef RationalQuadraticLift < KernelLift
    %RATIONALQUADRATICLIFT Rational quadratic kernel feature lift.

    properties (SetAccess=protected)
        lengthScale (1,1) {mustBePositive} = 1;
        alpha (1,1) {mustBePositive} = 1;
    end

    methods
        function obj = RationalQuadraticLift(dim, lengthScale, alpha, centers, options)
            arguments
                dim (1,1) {mustBeInteger, mustBePositive};
                lengthScale (1,1) {mustBePositive};
                alpha (1,1) {mustBePositive};
                centers (:,:) {mustBeNumeric};
                options.includeState (1,1) logical = false;
            end
            obj = obj@KernelLift(dim, centers, includeState=options.includeState);
            obj.lengthScale = lengthScale;
            obj.alpha = alpha;
        end
    end

    methods (Access=protected)
        function k = kernel(obj, x, y)
            k = (1 + sum((x - y).^2, 1) / (2 * obj.alpha * obj.lengthScale^2)) .^ (-obj.alpha);
        end
    end
end
