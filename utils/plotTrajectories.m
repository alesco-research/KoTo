function [fig, axs] = plotTrajectories(X, labels, options)
% PLOTTRAJECTORIES  Plot concatenated state and input trajectories in a
% stacked tiled layout, with run boundaries marked by vertical lines.
%
%   [fig, axs] = plotTrajectories(X1, labels1, X2, labels2, ..., Name=Value)
%
%   Inputs (repeating pairs):
%       Xi      - Signal trajectories, cell array of n_i-by-N_j-by-p_i
%                 arrays (one cell per run j). Each row gets its own tile;
%                 the p_i pages are drawn as separate lines in that tile.
%                 All Xi must contain the same number of runs.
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
%       colorscheme - n-by-3 RGB matrix used as color order
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
        options.name (1,1) string = "Trajectories"
        options.dt (1,1) {mustBeNumeric, mustBePositive} = 1
        options.axs (:,1) matlab.graphics.axis.Axes = matlab.graphics.axis.Axes.empty
        options.lineStyle (1,1) string = "-"
        options.colorscheme (:,3) {mustBeNumeric, mustBeBetween(options.colorscheme, 0, 1)} = [ ...
            0.00 0.45 0.74; ...
            0.00 0.60 0.20; ...
            0.85 0.10 0.10; ...
            0.75 0.00 0.75 ...
        ]
    end
    if isempty(X)
        error("Provide at least one trajectory to plot.");
    end
    if any(cellfun(@numel, X) ~= numel(X{1}))
        error("Inputs must have the same number of runs.");
    end

    nRows = cellfun(@(x) size(x{1}, 1), X);
    runLen = cellfun(@(x) cellfun(@(run) size(run, 2), x(:)), X, UniformOutput=0);
    runLen = max([runLen{:}], [], 2);
    runStart = [0; cumsum(runLen(1:end-1))];
    runBounds = options.dt * runStart(2:end);

    overlay = ~isempty(options.axs);
    if overlay && numel(options.axs) ~= sum(nRows)
        error("axs must contain %d axes, one per signal row.", sum(nRows));
    end

    for i = 1:numel(X)
        if isempty(labels{i})
            if ~overlay
                labels{i} = cellstr("Traj. " + i + "." + (1:nRows(i)));
            end
        elseif numel(labels{i}) ~= nRows(i)
            error("Labels for signal %d must contain %d entries.", i, nRows(i));
        end
    end

    if overlay
        axs = options.axs;
        fig = ancestor(axs(1), "figure");
    else
        fig = figure(Name=options.name);
        tl = tiledlayout(fig, sum(nRows), 1, TileSpacing="compact", Padding="compact");
        colororder(fig, options.colorscheme)
        axs = gobjects(sum(nRows), 1);
    end

    k = 0;
    for i = 1:numel(X)
        Xcat = [X{i}{:}];
        t = cellfun(@(run, t0) t0 + (0:size(run, 2) - 1), X{i}(:), num2cell(runStart), UniformOutput=false);
        t = options.dt * [t{:}];
        for row = 1:nRows(i)
            k = k + 1;
            if ~overlay
                axs(k) = nexttile(tl);
                hold(axs(k), "on");
                if size(Xcat, 3) > 1
                    axs(k).ColorOrderIndex = 1;
                else
                    axs(k).ColorOrderIndex = k;
                end
                % When overlaying, the color order continues where it left off
            end
            
            plot(axs(k), t, squeeze(Xcat(row, :, :)), options.lineStyle, LineWidth=1.5);
            if ~overlay && ~isempty(runBounds)
                xline(axs(k), runBounds, "r", LineWidth=0.5, HandleVisibility="off");
            end
            if ~isempty(labels{i})
                ylabel(axs(k), labels{i}{row}, Interpreter="latex");
            end
        end
    end
    linkaxes(axs, "x");
    xlabel(axs(end), "Time [s]");
    set(axs(1:end-1), "XTickLabel", []);
    xlim(axs(1), "tight");
end

function mustBeValidTrajectory(X)
    nRows  = cellfun(@(x) size(x, 1), X(:)');
    nPages = cellfun(@(x) size(x, 3), X(:)');
    if any(nRows ~= nRows(1)) || any(nPages ~= nPages(1))
        error("All runs must have the same number of rows and pages. " + ...
            "Instead has rows %s and pages %s", mat2str(nRows), mat2str(nPages));
    end
end
