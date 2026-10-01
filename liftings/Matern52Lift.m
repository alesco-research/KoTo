classdef Matern52Lift < KernelLift
    %MATERN52LIFT Matern 5/2 kernel feature lift.

    properties (SetAccess=protected)
        lengthScale (1,1) double {mustBePositive} = 1;
    end

    methods
        function obj = Matern52Lift(dim, lengthScale, centers, options)
            arguments
                dim (1,1) {mustBeInteger, mustBePositive};
                lengthScale (1,1) {mustBePositive};
                centers (:,:) {mustBeNumeric};
                options.includeState (1,1) logical = false;
            end
            obj = obj@KernelLift(dim, centers, includeState=options.includeState);
            obj.lengthScale = lengthScale;
        end
    end

    methods (Access=protected)
        function k = kernel(obj, x, y)
            r  = sqrt(sum((x - y).^2, 1));
            rs = sqrt(5) * r / obj.lengthScale;
            k  = (1 + rs + (5/3) * (r / obj.lengthScale).^2) .* exp(-rs);
        end
    end
end
