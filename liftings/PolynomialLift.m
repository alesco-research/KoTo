classdef PolynomialLift < LiftBase
    %POLYNOMIALLIFT Full multivariate polynomial basis up to a given
    %total degree, ordered with the state vector first (graded order).

    properties (SetAccess=protected)
        dim = 1;
        featureDim = 1;
        includeState = true;
        degree (1,1) {mustBeInteger, mustBePositive} = 1;
        exponents (:,:) {mustBeInteger, mustBeNonnegative} = [];
    end

    methods
        function obj = PolynomialLift(dim, degree)
            arguments
                dim (1,1) {mustBeInteger, mustBePositive};
                degree (1,1) {mustBeInteger, mustBePositive};
            end
            obj.dim = dim;
            obj.degree = degree;
            obj.exponents = obj.computeExponents(dim, degree);
            obj.featureDim = nchoosek(dim + degree, degree);
        end

        function Z = lift(obj, X)
            arguments
                obj (1,1) PolynomialLift;
                X (:,:) {mustBeNumeric, LiftBase.mustMatchDim(obj, X)};
            end
            Z = zeros(size(obj.exponents, 2), size(X, 2));
            for k = 1:size(obj.exponents, 2)
                Z(k,:) = prod(X.^obj.exponents(:,k), 1);
            end
        end
    end

    methods (Access=private)
        function exps = computeExponents(obj, dim, degree)
            exps = [];
            for i = 0:degree
                exps = [exps, obj.getCompositeExponents(dim, i)']; %#ok<AGROW>
            end
            exps = flip(exps, 1); % returns state vector first
        end

        function C = getCompositeExponents(obj, dim, degree)
            if dim == 1
                C = double(degree);
            else
                C = [];
                for i = 0:degree
                    rest = obj.getCompositeExponents(dim - 1, degree - i);
                    C = [C; double(i)*ones(size(rest,1), 1), rest]; %#ok<AGROW>
                end
            end
        end
    end
end
