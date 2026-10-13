classdef MOESP
%MOESP Subspace identification of a linear state-space model.
%   Identifies (A, B, C, D) from input-output data with the MOESP
%   algorithm and provides one-step prediction and warm-started forecasting.
%
%   obj = MOESP(Y, U, L, gamma=0.85, stateDim=[])
%     Y         ny-by-N output data (samples as columns).
%     U         nu-by-N input data (samples as columns).
%     L         Block-Hankel depth. Must be larger than the state dimension.
%     gamma     Cumulative singular-value threshold in (0, 1) used to select
%               the model order (default 0.85).
%     stateDim  Fixes the model order and overrides gamma.
%
%   Properties (read-only):
%     A, B, C, D                State-space matrices.
%     inputDim, outputDim       Number of inputs and outputs.
%     stateDim                  Identified model order.
%     sigma                     Singular values used for order selection.
%     memory                    Number of past samples used by forecast to
%                               estimate the initial state:
%                               2*ceil(stateDim/outputDim), which makes the
%                               estimation problem overdetermined.
%     O, G                      Observability matrix and Toeplitz matrix of
%                               Markov parameters over the memory window,
%                               used to estimate the initial state.
%
%   [x, y] = predict(obj, x0, u)
%     One-step prediction: x = A*x0 + B*u, y = C*x0 + D*u.
%
%   Y = forecast(obj, uPast, yPast, uFuture)
%     Warm-started multi-step output prediction.
%       uPast, yPast  Past input and output samples (columns). Both must
%                     have the same number of columns, at least `memory`.
%                     Only the most recent `memory` samples are used.
%       uFuture       nu-by-T future inputs.
%     The state at the start of the window is estimated by least squares
%     from the last `memory` samples of (uPast, yPast), then simulated
%     through that window and the future inputs. Returns an
%     ny-by-(memory+T) matrix Y, where the first `memory` columns (the
%     warm-up window) are NaN and columns memory+1:end hold the predicted
%     outputs for uFuture.

    properties (SetAccess=immutable)
        outputDim {mustBeInteger, mustBePositive} = [];
        stateDim {mustBeInteger, mustBePositive} = [];
        inputDim {mustBeInteger, mustBePositive} = [];
        A (:,:) {mustBeNumeric, mustBeMatrix};
        B (:,:) {mustBeNumeric, mustBeMatrix};
        C (:,:) {mustBeNumeric, mustBeMatrix};
        D (:,:) {mustBeNumeric, mustBeMatrix};
        O (:,:) {mustBeNumeric, mustBeMatrix};
        G (:,:) {mustBeNumeric, mustBeMatrix};
        sigma (:,1) {mustBeNumeric, mustBeNonnegative};
        memory {mustBeInteger, mustBePositive};
    end

    methods
        function obj = MOESP(Y, U, L, options)
            arguments
                Y (:,:) {mustBeNumeric}
                U (:,:) {mustBeNumeric, MOESP.mustBeEqualSize(Y, U)}
                L (1,1) {mustBeInteger, mustBePositive}
                options.gamma (1,1) {mustBeNumeric, mustBeBetween(options.gamma, 0, 1)} = 0.92;
                options.stateDim (:,:) {mustBeInteger, MOESP.mustBeValidLength(L, options.stateDim)} = [];
            end
            [ny, nu] = deal(size(Y, 1), size(U, 1));

            % Get scaled hankel matrices
            N = size(Y, 2) - L + 1;
            Hy = MOESP.hankel(Y, L) ./ sqrt(N);
            Hu = MOESP.hankel(U, L) ./ sqrt(N);

            Hyp = Hy - (Hy / Hu) * Hu;
            [Un, obj.sigma] = svd(Hyp, "vector");

            nx = options.stateDim;
            if isempty(nx)
                nx = find(cumsum(obj.sigma) / sum(obj.sigma) > options.gamma, 1);
                MOESP.mustBeValidLength(L, nx)
            end

            obsL = Un(:, 1:nx) * diag(sqrt(obj.sigma(1:nx)));

            % Compute A and B
            obj.C = obsL(1:ny, :);
            obj.A = pinv(obsL(1:ny * (L - 1),:)) * obsL(ny + 1:L * ny,:);
            
            % Compute B and D
            U2t = Un(:, nx + 1:end)';
            M1 = U2t * (Hy / Hu);
            m = ny * L - nx;
            M = zeros(m * L, nu);
            R = zeros(m * L, ny + nx);
            for k = 1:L
                M((k-1) * m + 1:k * m,:) = M1(:,(k - 1) * nu + 1:k * nu);
                R((k-1) * m + 1:k * m,:) = [
                    U2t(:, (k - 1) * ny + 1:k * ny), ...
                    U2t(:, k * ny + 1:end) * obsL(1:end - k * ny,:)
                ];
            end
            DB = R \ M;
            obj.D = DB(1:ny, :);
            obj.B = DB(ny + 1:end, :);
            obj.outputDim = ny;
            obj.stateDim = nx;
            obj.inputDim = nu;
            obj.memory = 2*ceil(nx/ny);
            [obj.O, obj.G] = obj.observabilityAndToeplitz(obj.memory);
        end

        function [x, y] = predict(obj, x0, u)
            arguments
                obj (1,1) MOESP;
                x0 (:,:) {MOESP.mustMatch(obj, "stateDim", x0)};
                u (:,:) {MOESP.mustMatch(obj, "inputDim", u)};
            end
            x = obj.A * x0 + obj.B * u;
            y = obj.C * x0 + obj.D * u;
        end

        function Y = forecast(obj, uPast, yPast, uFuture)
            arguments
                obj (1,1) MOESP;
                uPast (:,:) {MOESP.mustMatch(obj, "inputDim", uPast), MOESP.mustMatchMemory(obj, uPast)};
                yPast (:,:) {MOESP.mustMatch(obj, "outputDim", yPast), MOESP.mustBeEqualSize(yPast, uPast)};
                uFuture (:,:) {MOESP.mustMatch(obj, "inputDim", uFuture)};
            end
            yPast = yPast(:,1 + end - obj.memory:end);
            uPast = uPast(:,1 + end - obj.memory:end);
            T = size(uFuture, 2);
            
            % Estimate x0
            x = obj.O \ (yPast(:) - obj.G * uPast(:));
            
            u = [uPast, uFuture];
            Y = zeros(obj.outputDim, obj.memory + T);
            for t = 1:(obj.memory + T)
                [x, Y(:, t)] = obj.predict(x, u(:, t));
            end
            Y(:, 1:obj.memory) = nan;
        end
    end

    methods (Access=protected)
        function [O, G] = observabilityAndToeplitz(obj, m)
            ny = obj.outputDim;
            nx = obj.stateDim;

            O = zeros(m * ny, nx);      % observability matrix
            G = kron(eye(m), obj.D);    % toeplitz matrix

            CAk = obj.C;
            for k = 1:(m-1)
                O((k - 1) * ny + (1:ny), :) = CAk;
                G = G + kron(diag(ones(m - k,1), -k), CAk * obj.B);
                CAk = CAk * obj.A;
            end
            O((m - 1) * ny + (1:ny), :) = CAk;
        end
    end

    methods (Static, Hidden, Access=protected)
        function h = hankel(X, L)
            [d, T] = size(X);
            cols = T - L + 1;

            idx = (0:L-1)' + (1:cols);
            h = reshape(X(:, idx(:)), L * d, cols);
        end

        function mustBeValidLength(L, stateDim)
            if L <= stateDim
                error( ...
                    "MOESP:HankelMatrixTooShort", ...
                    "L %d is too small. Must be larger than the " + ...
                    "state dimension %d.", ...
                    L, stateDim ...    
                )
            end
        end

        function mustBeEqualSize(Y, U)
            if size(Y, 2) ~= size(U, 2)
                error( ...
                    "MOESP:DimensionMismatch", ...
                    "Inputs must have the same number of samples. " + ...
                    "Received inputs of dimension %d and %d instead.", ...
                    size(Y, 2), ...
                    size(U, 2) ...
            );
            end
        end

        function mustMatch(obj, dim, val)
            if size(val, 1) ~= obj.(dim)
                error( ...
                    "MOESP:DimensionMismatch", ...
                    "Value must have %d row(s) to match the " + ... 
                    "%s dimension.", ...
                    obj.(dim), erase(dim, "Dim") ...
                );
            end
        end

        function mustMatchMemory(obj, val)
            if size(val, 2) < obj.memory
                error( ...
                    "MOESP:MemoryMismatch", ...
                    "Must provide %d past inputs and outputs to match the " + ... 
                    "memory structure of the predictor.", ...
                    obj.memory ...
                );
            end
        end
    end
end