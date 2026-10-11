function [fig, axs] = plotTrajectories(X, U, dt, options)
% PLOTTRAJECTORIES  Plot concatenated state and input trajectories in a
% stacked tiled layout, with trajectory boundaries marked by vertical lines.
%
%   [fig, axs] = plotTrajectories(X, U, dt, Name=Value)
%
%   Inputs:
%       X  - State trajectories, cell array of nx-by-N_i matrices
%       U  - Input trajectories, cell array of nu-by-(N_i-1) matrices
%       dt - Sampling time in seconds (default: 1)
%
%   Name-Value Options:
%       name        - Figure name (default: "Trajectories")
%       stateLabels - Cell array with one y-label per state; use a string
%                     array for multi-line labels (default: "State i")
%       inputLabels - Same as stateLabels, for inputs (default: "Input i")
%       colorscheme - n-by-3 RGB matrix, states first, then inputs
%
%   Outputs:
%       fig - Figure handle
%       axs - Handles to the (nx+nu) axes, top to bottom
%
%   Example:
%       plotTrajectories(XTrain, UTrain, dt, ...
%           stateLabels={["Temp. 1" "$[^\circ$C]"], "Temp. 2"});
%
%   See also: tiledlayout, plot, stairs, xline
    arguments
        X (:,:) cell {mustBeValidTrajectory}
        U (:,:) cell {mustBeValidTrajectory, mustBeValidInput(X,U)}
        dt (1,1) {mustBeNumeric, mustBePositive} = 1;
        options.name (1,1) string = "Trajectories"
        options.stateLabels (:,:) cell = {}
        options.inputLabels (:,:) cell = {}
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
    [xDim, uDim] = deal(size(X{1}, 1), size(U{1}, 1));
    [Xall, Uall] = deal([X{:}], [U{:}]);
    
    lastIdx = cumsum(cellfun(@(x) size(x, 2), X(:)))';
    
    timeX = (0:size(Xall, 2) - 1) * dt;
    timeU = timeX;
    timeU(lastIdx) = [];
    separators = timeX(1 + lastIdx(1:end-1));

    if isempty(options.stateLabels)
        options.stateLabels = cellstr("State " + (1:xDim));
    elseif numel(options.stateLabels) ~= xDim
        error("stateLabels must contain %d entries.", xDim);
    end
    if isempty(options.inputLabels)
        options.inputLabels = cellstr("Input " + (1:uDim));
    elseif numel(options.inputLabels) ~= uDim
        error("inputLabels must contain %d entries.", uDim);
    end

    fig = figure(Name=options.name);
    tl  = tiledlayout( ...
        fig, xDim + uDim, 1, TileSpacing="compact", Padding="compact" ...
    );

    colororder(options.colorscheme)
    axs = gobjects(xDim + uDim, 1);
    % Plot states
    for i = 1:xDim
        axs(i) = nexttile(tl);
        
        hold(axs(i), "on");
        axs(i).ColorOrderIndex = i;
        plot(axs(i), timeX, Xall(i,:), LineWidth=1.5);
        if ~isempty(separators)
            xline(axs(i), separators, "r", LineWidth=0.5);
        end
        ylabel(axs(i), options.stateLabels{i}, Interpreter="latex");
    end
    % Plot inputs
    for i = xDim + (1:uDim)
        axs(i) = nexttile(tl);
        
        hold(axs(i), "on");
        axs(i).ColorOrderIndex = i;
        stairs(axs(i), timeU, Uall(i - xDim, :), LineWidth=1.5);
        if ~isempty(separators)
            xline(axs(i), separators, "r", LineWidth=0.5);
        end
        ylabel(axs(i), options.inputLabels{i - xDim}, Interpreter="latex");
    end

    linkaxes(axs, "x");
    xlabel(axs(end), "Time [s]");
    set(axs(1:end-1), "XTickLabel", []);
    xlim(axs(1), [timeX(1) timeX(end)]);
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
end

function mustBeValidInput(X, U)
    if numel(X) ~= numel(U)
        error("X and U must have the same number of trajectories.");
    end
    if ~all(cellfun(@(x, u) size(x, 2) - 1 == size(u, 2), X(:), U(:)))
        error("X and U must have trajectories of matching length.");
    end
end