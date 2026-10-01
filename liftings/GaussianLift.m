classdef GaussianLift < KernelLift
    %GAUSSIANLIFT Gaussian RBF kernel feature lift.

    properties (SetAccess=protected)
        sigma (1,1) {mustBePositive} = 1;
    end

    methods
        function obj = GaussianLift(dim, sigma, centers, options)
            arguments
                dim (1,1) {mustBeInteger, mustBePositive};
                sigma (1,1) {mustBePositive};
                centers (:,:) {mustBeNumeric};
                options.includeState (1,1) logical = false;
            end
            obj = obj@KernelLift(dim, centers, includeState=options.includeState);
            obj.sigma = sigma;
        end
    end

    methods (Access=protected)
        function k = kernel(obj, x, y)
            k = exp(-sum((x - y).^2, 1) / (2 * obj.sigma^2));
        end
    end
end
