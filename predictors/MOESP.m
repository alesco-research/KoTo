classdef MOESP
%MOESP Subspace identification of a linear state-space model.
%   Identifies (A, B, C, D) from input-output data with the MOESP
%   algorithm and provides one-step prediction and warm-started forecasting.
%
%   obj = MOESP(Y, U, L, gamma=0.9, stateDim=[])
%     Y         Cell array of ny-by-N_i output trajectories.
%     U         Cell array of nu-by-N_i input trajectories, matching Y.
%               For a single trajectory use MOESP({Y1}, {U1}, L).
%     L         Block-Hankel depth. Must exceed the state dimension, and
%               every trajectory needs at least L samples.
%     gamma     Cumulative singular-value threshold in (0, 1) used to select
%               the model order (default 0.9).
%     stateDim  Fixes the model order and overrides gamma.
%
%   Hankel matrices are built per trajectory and concatenated column-wise,
%   so trajectories may differ in length and start from different states.
%
%   Properties (read-only):
%     A, B, C, D                State-space matrices.
%     inputDim, outputDim       Number of inputs and outputs.
%     stateDim                  Identified model order.
%     sigma                     Singular values used for order selection.
%     memory                    Past samples used by forecast to estimate
%                               the initial state: 2*ceil(stateDim/outputDim).
%     O, G                      Observability and Toeplitz matrices over the
%                               memory window, used to estimate the initial
%                               state.
%
%   [x, y] = predict(obj, x0, u)
%     One-step prediction: x = A*x0 + B*u, y = C*x0 + D*u.
%
%   Y = forecast(obj, uPast, yPast, uFuture)
%     Warm-started multi-step output prediction.
%       uPast, yPast  Past samples (columns), at least `memory` each, ending
%                     at the same time step. Only the last `memory` are used.
%       uFuture       nu-by-T future inputs.
%     The initial state is estimated by least squares from the past window.
%     Returns an ny-by-(memory+T) matrix; the first `memory` columns are NaN
%     and the rest are the predicted outputs for uFuture.

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
                Y (:,:) cell {mustBeVector};
                U (:,:) cell {mustBeVector, MOESP.mustBeEqualSize(Y, U)}
                L (1,1) {mustBeInteger, mustBePositive}
                options.gamma (1,1) {mustBeNumeric, mustBeBetween(options.gamma, 0, 1)} = 0.9;
                options.stateDim (:,:) {mustBeInteger, MOESP.mustBeValidLength(L, options.stateDim)} = [];
            end
            [ny, nu] = deal(size(Y{1}, 1), size(U{1}, 1));

            % Per-trajectory Hankel matrices, concatenated column-wise
            Hy = cellfun(@(y) MOESP.hankel(y, L), Y, UniformOutput=false);
            Hu = cellfun(@(u) MOESP.hankel(u, L), U, UniformOutput=false);
            Hy = [Hy{:}];
            Hu = [Hu{:}];

            % Scale by the total number of columns
            N = size(Hy, 2);
            Hy = Hy ./ sqrt(N);
            Hu = Hu ./ sqrt(N);

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
                yPast (:,:) {MOESP.mustMatch(obj, "outputDim", yPast), MOESP.mustMatchMemory(obj, yPast)};
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
            Y(:, 1:obj.memory) = [];
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

            if cols < 1
                error( ...
                    "MOESP:TrajectoryTooShort", ...
                    "Trajectory has %d samples but needs at least L = %d.", ...
                    T, L ...
                );
            end

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
            if numel(Y) ~= numel(U)
                error( ...
                    "MOESP:DimensionMismatch", ...
                    "Y and U must have an equal number of trajectories." ...
                );
            end
            if any(cellfun(@(y, u) size(y, 2) ~= size(u, 2), Y(:), U(:)))
                error( ...
                    "MOESP:DimensionMismatch", ...
                    "Y and U must have trajectories of equal length." ...
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