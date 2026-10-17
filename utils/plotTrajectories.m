function [fig, axs] = plotTrajectories(X, trajOpts, options)
% PLOTTRAJECTORIES  Plot concatenated state and input trajectories in a
% stacked tiled layout, with run boundaries marked by vertical lines.
%
%   [fig, axs] = plotTrajectories(X1, opts1, X2, opts2, ..., Name=Value)
%
%   Inputs (repeating pairs):
%       Xi    - Signal trajectories, cell array of n_i-by-N_j-by-p_i
%               arrays (one cell per run j). Each row gets its own tile;
%               the p_i pages are drawn as separate lines in that tile.
%               All Xi must contain the same number of runs.
%       optsi - Per-trajectory options struct, see trajectoryOptions for
%               the available fields. Missing fields take their defaults,
%               so struct() gives an all-default trajectory.
%
%   Name-Value Options:
%       dt          - Sampling time in seconds (default: 1)
%       name        - Figure name (default: "Trajectories")
%       axs         - Axes from a previous call to draw on top of; then no
%                     new figure is created and name/colorscheme/linestyles
%                     are ignored. Labels may be {} to keep the existing ones.
%       colorscheme - n-by-3 RGB matrix or palette name used as color order
%                     (default: "gem12")
%       linestyles  - Line styles or style-order name used as line style
%                     order, cycled together with the colors
%                     (default: "mixedstyles")
%
%   Outputs:
%       fig - Figure handle
%       axs - Handles to the sum(n_i) axes, top to bottom
%
%   Example:
%       plotTrajectories( ...
%           XTrain, trajectoryOptions(labels={["Temp. 1" "$[^\circ$C]"]; "Temp. 2"}), ...
%           UTrain, struct(), dt=Ts);
%
%       [~, axs] = plotTrajectories(XTest, struct(), dt=Ts);
%       plotTrajectories(XPred, trajectoryOptions(lineStyle="--"), dt=Ts, axs=axs);
%
%   See also: trajectoryOptions, tiledlayout, plot, xline
    arguments (Repeating)
        X (:,:) cell {mustBeVector, mustBeValidTrajectory}
        trajOpts (1,1) struct
    end
    arguments
        options.name (1,1) string = "Trajectories"
        options.dt (1,1) {mustBeNumeric, mustBePositive} = 1
        options.colorscheme (:,:) {mustBeValidColorOrder} = "gem12"
        options.linestyles (:,:) {mustBeValidLineStyleOrder} = "mixedstyles"
        options.axs (:,1) matlab.graphics.axis.Axes = matlab.graphics.axis.Axes.empty
    end
    if isempty(X)
        error("Provide at least one trajectory to plot.");
    end
    if any(cellfun(@numel, X) ~= numel(X{1}))
        error("Inputs must have the same number of runs.");
    end
    for i = 1:numel(trajOpts)
        try
            args = namedargs2cell(trajOpts{i});
            trajOpts{i} = trajectoryOptions(args{:});
        catch e
            error("Invalid options for trajectory %d: %s", i, e.message);
        end
    end

    % Get run lengths
    nRows = cellfun(@(x) size(x{1}, 1), X);
    runLen = cellfun(@(x) cellfun(@(run) size(run, 2), x(:)), X, UniformOutput=0);
    runLen = max([runLen{:}], [], 2);
    runStart = [0; cumsum(runLen(1:end-1))];
    runBounds = options.dt * runStart(2:end);

    % Check axs
    overlay = ~isempty(options.axs);
    if overlay && numel(options.axs) ~= sum(nRows)
        error("axs must contain %d axes, one per signal row.", sum(nRows));
    end

    % Check labels and displayName
    for i = 1:numel(X)
        if ~isempty(trajOpts{i}.labels) && numel(trajOpts{i}.labels) ~= nRows(i)
            error("Labels for signal %d must contain %d entries.", i, nRows(i));
        end
        nPages = size(X{i}{1}, 3);
        if ~isempty(trajOpts{i}.displayName) && numel(trajOpts{i}.displayName) ~= nPages
            error("displayName for signal %d must contain %d entries.", i, nPages);
        end
    end

    if overlay
        axs = options.axs;
        fig = ancestor(axs(1), "figure");
    else
        fig = figure(Name=options.name);
        tl = tiledlayout(fig, sum(nRows), 1, TileSpacing="compact", Padding="compact");
        axs = arrayfun(@(x) nexttile(tl), 1:sum(nRows));
        linestyleorder(fig, options.linestyles)
        colororder(fig, options.colorscheme)
    end

    j = 0;
    for i = 1:numel(X)
        Xcat = [X{i}{:}];
        t = cellfun( ...
            @(run, t0) t0 + (0:size(run, 2) - 1), X{i}(:), num2cell(runStart), ...
            UniformOutput=false ...
        );
        t = options.dt * [t{:}];
        for row = 1:nRows(i)
            j = j + 1;
            hold(axs(j), "on");
            if ~overlay && size(Xcat, 3) == 1
                axs(j).ColorOrderIndex = j;
                axs(j).LineStyleOrderIndex = j;
            end
            
            hLines = plot( ...
                axs(j), t, squeeze(Xcat(row, :, :)), LineWidth=trajOpts{i}.lineWidth ...
            );
            if ~isempty(trajOpts{i}.lineStyle)
                set(hLines, "LineStyle", trajOpts{i}.lineStyle);
            end
            if ~isempty(trajOpts{i}.color)
                set(hLines, "Color", trajOpts{i}.color);
            end
            if ~isempty(trajOpts{i}.displayName)
                set(hLines, {"DisplayName"}, cellstr(trajOpts{i}.displayName));
            end
            if ~overlay && ~isempty(runBounds)
                xline(axs(j), runBounds, "r", LineWidth=0.5, HandleVisibility="off");
            end
            if ~isempty(trajOpts{i}.labels)
                ylabel(axs(j), trajOpts{i}.labels{row});
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
        error( ...
            "All runs must have the same number of rows and pages. " + ...
            "Instead has rows %s and pages %s", ...
            mat2str(nRows), mat2str(nPages) ...
        );
    end
end

function mustBeValidColorOrder(colors)
    fig = figure(Visible="off");
    try
        colororder(fig, colors);
        close(fig);
    catch e
        close(fig);
        error(e.message);
    end
end

function mustBeValidLineStyleOrder(lineStyles)
    fig = figure(Visible="off");
    try
        linestyleorder(fig, lineStyles);
        close(fig);
    catch e
        close(fig);
        error(e.message);
    end
end