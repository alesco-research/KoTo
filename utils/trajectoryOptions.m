function opts = trajectoryOptions(options)
% TRAJECTORYOPTIONS  Create and validate per-trajectory plot options for
% plotTrajectories.
%
%   opts = trajectoryOptions(Name=Value)
%
%   Name-Value Options:
%       labels      - Cell array with one y-label per row; use a string
%                     array for multi-line labels, or {} for defaults
%                     (default: {})
%       lineStyle   - Line style, one of "-", "--", ":", "-.", "none", or
%                     empty to use the axes LineStyleOrder
%                     (default: string.empty)
%       color       - Line color as RGB triplet, hex code or color name
%                     (e.g. "k"), or empty to use the axes ColorOrder
%                     (default: [])
%       lineWidth   - Line width passed to plot, or empty to use the
%                     default LineWidth (default: [])
%       displayName - Legend entries, one per page of the trajectory, or
%                     empty to leave them unset (default: string.empty)
%
%   Output:
%       opts - Scalar struct with all fields set
%
%   An existing struct (e.g. with only some fields set) can be validated
%   and completed with defaults via
%       args = namedargs2cell(s);
%       opts = trajectoryOptions(args{:});
%
%   Example:
%       plotTrajectories(XPred, trajectoryOptions(lineStyle="--"), dt=Ts);
%
%   See also: plotTrajectories, validatecolor
    arguments
        options.labels (:,:) cell = {}
        options.lineStyle (:,1) string { ...
            mustBeMember(options.lineStyle, ["-", "--", ":", "-.", "none"]) ...
        } = string.empty
        options.color {mustBeValidColor} = []
        options.lineWidth double {mustBeScalarOrEmpty, mustBePositive} = []
        options.displayName (:,1) string = string.empty
    end
    opts = options;
    if ~isempty(opts.color)
        opts.color = validatecolor(opts.color);
    end
end

function mustBeValidColor(color)
    if ~isempty(color)
        validatecolor(color);
    end
end
