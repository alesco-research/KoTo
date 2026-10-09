function exportPlot(fig, filename, options)
% EXPORTPLOT  Export a figure to file with control over size, margins
% and font sizes.
%
%   exportPlot(fig, filename) exports the figure fig to the file specified
%   by filename, using the current figure dimensions and default settings.
%
%   exportPlot(fig, filename, Name=Value) applies additional formatting
%   before export using the name-value options described below.
%
%   Input Arguments:
%       fig      - Figure to export (matlab.ui.Figure)
%       filename - Output file path, including extension (string)
%                  The file format is inferred from the extension unless
%                  'formattype' is specified.
%
%   Name-Value Options:
%       formattype     - Format string passed to saveas(), e.g. 'png',
%                        'pdf', 'epsc'. Inferred from filename if omitted.
%                        "" (default) | string
%       width          - Output figure width in 'unit' units.
%                        current figure width (default) | numeric
%       height         - Output figure height in 'unit' units.
%                        current figure height (default) | numeric
%       margin         - Margin(s) between paper edge and axes area, in
%                        'unit' units. Accepts 1, 2, or 4 elements:
%                        1 element  – uniform margin on all sides [m m m m]
%                        2 elements – [left/right, top/bottom]
%                        4 elements – [left, right, top, bottom]
%                        1 (default) | numeric
%       labelFontSize  - Font size for all axis labels.
%                        Unchanged if omitted.
%                        numeric
%       tickFontSize   - Font size for all axis tick. Unchanged if omitted.
%                        numeric
%       legendFontSize - Font size for all legends in the figure.
%                        Unchanged if omitted. (numeric)
%       unit           - Unit used for width, height, and margin.
%                        'centimeters' (default) |'inches' |'points'
%
%   Examples:
%       % Basic export, format inferred from extension
%       exportPlot(gcf, 'results/myplot.png');
%
%       % Export as PDF at a fixed size with a 0.5 cm uniform margin
%       exportPlot(gcf, 'report/figure1.pdf', ...
%           formattype='pdf', width=12, height=8, margin=0.5);
%
%       % Export with custom font sizes and asymmetric margins
%       exportPlot(gcf, 'output/fig.eps', formattype='epsc', ...
%           width=16, height=10, margin=[1.5 1 1.5 0.8], ...
%           labelFontSize=11, tickFontSize=9, legendFontSize=9);
%
%   See also: saveas, print, exportgraphics
    arguments
        fig matlab.ui.Figure
        filename string
        options.formattype string = ""
        options.width {mustBeNumeric} = NaN
        options.height {mustBeNumeric} = NaN
        options.margin {mustBeNumeric} = 1
        options.labelFontSize {mustBeNumeric} = NaN
        options.tickFontSize {mustBeNumeric} = NaN
        options.legendFontSize {mustBeNumeric} = NaN
        options.unit string {mustBeValidUnit(options.unit)} = 'centimeters'
    end

    % Change figure and paper units
    set(fig, Units=options.unit);
    set(fig, PaperUnits=options.unit);

    % Set defaults
    pos = get(fig, 'Position');
    if isnan(options.width);  options.width=pos(3);  end
    if isnan(options.height); options.height=pos(4); end
    switch numel(options.margin)
        case 1,  options.margin = repelem(options.margin, 4);
        case 2,  options.margin = repelem(options.margin, 2);
        case 3,  error('1, 2, or 4 elements only');
    end

    % Set font sizes
    axHandles  = findall(fig,'type','axes');
    legendHandles = findall(fig,'type','legend');
    if ~isnan(options.labelFontSize)
        for i = 1:numel(axHandles)
            currentAxis = axHandles(i);
            set(currentAxis.XLabel, FontSize=options.labelFontSize)
            set(currentAxis.YLabel, FontSize=options.labelFontSize)
            set(currentAxis.ZLabel, FontSize=options.labelFontSize)
        end
    end
    if ~isnan(options.tickFontSize)
        for i = 1:numel(axHandles)
            currentAxis = axHandles(i);
            set(currentAxis.XAxis, FontSize=options.tickFontSize)
            set(currentAxis.YAxis, FontSize=options.tickFontSize)
            set(currentAxis.ZAxis, FontSize=options.tickFontSize)
        end
    end
    if ~isnan(options.legendFontSize)
        for i = 1:numel(legendHandles)
            set(legendHandles(i), FontSize=options.legendFontSize)
        end
    end
    
    % Update width and height
    set(fig, Position=[pos(1), pos(2), options.width options.height]);
    set(fig, PaperSize=[options.width options.height]);
    set(fig, PaperPosition=[ ...
        options.margin(1) ...
        options.margin(4), ...
        options.width-sum(options.margin(1:2)) ...
        options.height-sum(options.margin(3:4)) ...
    ]);

    % Export plot
    if options.formattype==""
        saveas(fig, filename);
    else
        saveas(fig, filename, options.formattype);
    end
end

function mustBeValidUnit(val)
    mustBeMember(val, {'centimeters','inches','points'})
end