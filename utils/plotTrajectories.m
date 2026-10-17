function [fig, axs] = plotTrajectories(X, labels, options)
% PLOTTRAJECTORIES  Plot concatenated state and input trajectories in a
% stacked tiled layout, with trajectory boundaries marked by vertical lines.
%
%   [fig, axs] = plotTrajectories(X1, labels1, X2, labels2, ..., Name=Value)
%
%   Inputs (repeating pairs):
%       Xi      - Signal trajectories, cell array of n_i-by-N_j-by-p_i
%                 arrays (one cell per run). Each row gets its own tile;
%                 the p_i pages are drawn as separate lines in that tile.
%       labelsi - Cell array with one y-label per row; use a string array
%                 for multi-line labels, or {} for defaults
%
%   Name-Value Options:
%       dt          - Sampling time in seconds (default: 1)
%       name        - Figure name (default: "Trajectories")
%       axs         - Axes from a previous call to draw on top of; then no
%                     new figure is created and name/colorscheme are
%                     ignored. Labels may be {} to keep the existing ones.
%       lineStyle   - Line style passed to plot (default: "-")
%       colorscheme - n-by-3 RGB matrix
%
%   Outputs:
%       fig - Figure handle
%       axs - Handles to the sum(n_i) axes, top to bottom
%
%   Example:
%       plotTrajectories( ...
%           XTrain, {["Temp. 1" "$[^\circ$C]"], "Temp. 2"}, ...
%           UTrain, {}, dt=Ts);
%
%       [~, axs] = plotTrajectories(XTest, {}, dt=Ts);
%       plotTrajectories(XPred, {}, dt=Ts, axs=axs, lineStyle="--");
%
%   See also: tiledlayout, plot, xline
    arguments (Repeating)
        X (:,:) cell {mustBeVector, mustBeValidTrajectory}
        labels (:,:) cell
    end
    arguments
        options.dt (1,1) {mustBeNumeric, mustBePositive} = 1;
        options.name (1,1) string = "Trajectories"
        options.axs (:,1) matlab.graphics.axis.Axes = matlab.graphics.axis.Axes.empty
        options.lineStyle (1,1) string = "-"
        options.colorscheme (:,3) { ...
            mustBeNumeric, ...
            mustBeBetween(options.colorscheme, 0, 1, "closed") ...
            } = [ ...
            0.00 0.45 0.74; ...
            0.00 0.60 0.20; ...
            0.85 0.10 0.10; ...
            0.75 0.00 0.75 ...
        ];
    end
    if isempty(X)
        error("Provide at least one trajectory to plot.");
    end
    cellfun(@(x) mustBeCompatible(X{1}, x), X);

    dt = options.dt;
    dims = cellfun(@(x) size(x{1}, 1), X);
    
    Xall = cellfun(@(x) [x{:}], X, UniformOutput=false);
    steps = cellfun(@(Y) cellfun(@(y) size(y, 2), Y(:)), X, UniformOutput=0);
    
    segLen = max([steps{:}], [], 2);
    segStart = [0; cumsum(segLen(1:end-1))];
    separators = dt * segStart(2:end);
    times = arrayfun(@(i) dt * (segStart(i) + (0:segLen(i) - 1)), (1:numel(segLen))', UniformOutput=0);

    overlay = ~isempty(options.axs);
    if overlay && numel(options.axs) ~= sum(dims)
        error("axs must contain %d axes, one per signal row.", sum(dims));
    end

    for i = 1:numel(labels)
        if isempty(labels{i})
            if ~overlay
                labels{i} = cellstr("Traj. " + num2str(i) + "." + (1:dims(i)));
            end
        elseif numel(labels{i}) ~= dims(i)
            error("Labels for signal %d must contain %d entries.", i, dims(i));
        end
    end

    if overlay
        axs = options.axs(:);
        fig = ancestor(axs(1), "figure");
    else
        fig = figure(Name=options.name);
        tl  = tiledlayout( ...
            fig, sum(dims), 1, TileSpacing="compact", Padding="compact" ...
        );
        colororder(options.colorscheme)
        axs = gobjects(sum(dims), 1);
    end

    idx = 0;
    for i = 1:numel(dims)
        time = cellfun(@(t,x) t(1:size(x, 2)), times, X{i}(:), UniformOutput=0);
        for row = 1:dims(i)
            idx = idx + 1;
            if ~overlay
                axs(idx) = nexttile(tl);
                hold(axs(idx), "on");
                % Pages (3rd dim) become separate lines; restart colors so
                % that each page keeps its color across tiles
                if size(Xall{i}, 3) > 1
                    axs(idx).ColorOrderIndex = 1;
                else
                    axs(idx).ColorOrderIndex = idx;
                end
            end
            % When overlaying, the color order continues where it left off
            Y = reshape(Xall{i}(row, :, :), [], size(Xall{i}, 3));
            plot(axs(idx), [time{:}], Y, options.lineStyle, LineWidth=1.5);
            if ~overlay && ~isempty(separators)
                xline(axs(idx), separators, "r", LineWidth=0.5, HandleVisibility="off");
            end
            if ~isempty(labels{i})
                ylabel(axs(idx), labels{i}{row}, Interpreter="latex");
            end
        end
    end
    linkaxes(axs, "x");
    xlabel(axs(end), "Time [s]");
    set(axs(1:end-1), "XTickLabel", []);
    xlim(axs(1), "tight");
end

function mustBeValidTrajectory(X)
    dims = cellfun(@(x) size(x, 1), X(:));
    if any(dims ~= dims(1))
        error( ...
            "Trajectories must have consistent dimension across runs. " + ...
            "Instead has %s", ...
            mat2str(dims) ...
        );
    end
    pages = cellfun(@(x) size(x, 3), X(:));
    if any(pages ~= pages(1))
        error( ...
            "Trajectories must have a consistent number of pages across runs. " + ...
            "Instead has %s", ...
            mat2str(pages) ...
        );
    end
end

function mustBeCompatible(X, Y)
    if numel(X) ~= numel(Y)
        error("Inputs must have the same number of trajectories.");
    end
end