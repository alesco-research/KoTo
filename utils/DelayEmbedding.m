classdef DelayEmbedding
    %DELAYEMBEDDING Time-delay (Hankel) embedding of multivariate time series.
    %   obj = DELAYEMBEDDING(degree) stacks degree copies of the data,
    %   each delayed by one sample.
    %
    %   obj = DELAYEMBEDDING(degree, stride) delays successive copies by
    %   stride samples.
    %
    %   E = embed(obj, X)
    %       X is dim-by-T (columns are observations, rows are dimensions).
    %       E is (degree*dim)-by-(T-memory+1). The column belonging to time
    %       t is
    %           [X(:,t); X(:,t-stride); ...; X(:,t-(degree-1)*stride)],
    %       for t = memory, ..., T. The newest sample is the top block, so
    %       the current state is E(1:dim,:). X must have at least memory
    %       columns. Embed each experiment separately, never across
    %       experiment boundaries.
    %
    %   X = unembed(obj, E, dim)
    %       Return the current-sample block E(1:dim,:), i.e. the samples
    %       at times memory, ..., T. E must have degree*dim rows.
    %
    %   X = unembed(obj, E, dim, true)
    %       Also reconstruct and prepend the first memory-1 samples from
    %       the lagged blocks of E, recovering all T columns exactly (no
    %       interpolation). E must have at least stride columns.
    %
    %   Example:
    %       X  = randn(3, 50);
    %       de = DelayEmbedding(4, 2);
    %       E  = embed(de, X);               % 12-by-43
    %       Y  = unembed(de, E, 3, true);    % equals X
    %
    %   Properties (read-only):
    %      degree - Number of stacked copies
    %      stride - Delay between copies in samples (default 1)
    %      memory - Samples spanned by one column, stride*(degree-1) + 1

    properties (GetAccess=public, SetAccess=immutable)
        memory {mustBeInteger, mustBePositive}
        stride {mustBeInteger, mustBePositive}
        degree {mustBeInteger, mustBePositive}
    end

    methods
        function obj = DelayEmbedding(degree, stride)
            arguments
                degree (1,1);
                stride (1,1) = 1;
            end
            obj.degree = degree;
            obj.stride = stride;
            obj.memory = stride * (degree - 1) + 1;
        end

        function embedding = embed(obj, data)
            arguments
                obj (1,1) DelayEmbedding
                data (:,:) {mustBeNumeric, DelayEmbedding.mustBeEmbeddable(obj, data)}
            end
            [dim, T] = size(data);
            
            embedding = zeros(obj.degree * dim, T - obj.memory + 1);
            for i = 0:obj.degree-1
                cols = (obj.memory:T) - i * obj.stride;
                embedding(i * dim + (1:dim), :) = data(:, cols);
            end
        end

        function data = unembed(obj, embedding, dim, reconstruct)
            arguments
                obj (1,1) DelayEmbedding
                embedding (:,:) {mustBeNumeric}
                dim (1,1) {mustBeInteger, DelayEmbedding.mustBeEmbedding(obj, embedding, dim)}
                reconstruct (1,1) logical {DelayEmbedding.mustBeReconstructable(obj, embedding, reconstruct)} = false;
            end
            data = embedding(1:dim,:);
            if ~reconstruct || obj.degree == 1
                return                
            end
            
            H = embedding(dim + 1:end, 1:obj.stride);
            H = reshape(H, dim, obj.degree - 1, obj.stride);
            H = flip(H, 2);
            H = permute(H, [1 3 2]);
            hist = reshape(H, dim, []);
            data = [hist, data];
        end
    end

    methods (Access=private, Hidden, Static)
        function mustBeEmbeddable(obj, val)
            if size(val,2) < obj.memory
                error( ...
                    "DelayEmbedding:TooFewData", ...
                    "Data must have at least %d column(s) to be embedded.", ...
                    obj.memory ...
                );
            end
        end

        function mustBeEmbedding(obj, embedding, dim)
            if size(embedding,1) ~= obj.degree*dim
                error( ...
                    "DelayEmbedding:DimensionMismatch", ...
                    "Row count must equal degree*dim %d*%d.", ...
                    obj.degree, dim ...
                );
            end
        end

        function mustBeReconstructable(obj, embedding, flag)
            if flag && (size(embedding,2) < obj.stride)
                error( ...
                    "DelayEmbedding:TooFewColumns", ...
                    "Need at least stride = %d columns to reconstruct " + ...
                    "the history.", ...
                    obj.stride ...
                );
            end
        end
    end
end