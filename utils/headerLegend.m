function headerLegend(lgd, options)
% HEADERLEGEND  Center a legend in the upper or lower figure margin at the
% final export size.
%
%   headerLegend(lgd, Name=Value) resizes the figure containing the legend
%   lgd to its final export size, applies the legend formatting and then
%   centers the legend horizontally at the top or bottom edge of the
%   figure. Call it right before exportPlot with the same width, height
%   and unit, since the legend size depends on the figure size and font.
%
%   Input Arguments:
%       lgd - Legend to place (matlab.graphics.illustration.Legend)
%
%   Name-Value Options:
%       location   - Figure edge to place the legend at
%                    "north" (default) | "south"
%       width      - Final figure width in 'unit' units.
%                    current figure width (default) | numeric
%       height     - Final figure height in 'unit' units.
%                    current figure height (default) | numeric
%       unit       - Unit used for width and height.
%                    'centimeters' (default) |'inches' |'points'
%       fontSize   - Legend font size. Unchanged if omitted. (numeric)
%       lineLength - Length of the line in legend entries, in points.
%                    Unchanged if omitted. (numeric)
%       shrink     - Shrink the remaining figure content by the legend
%                    height to avoid overlap. false (default) | logical
%       gap        - Space between legend and shrunk content, in points.
%                    2 (default) | numeric
%
%   Example:
%       lgd = legend(axs(1), names, Orientation="horizontal");
%       headerLegend(lgd, width=8.6, height=5, fontSize=6, shrink=true);
%       exportPlot(gcf(), "plot.pdf", width=8.6, height=5, legendFontSize=6);
%
%   See also: exportPlot, legend
    arguments
        lgd matlab.graphics.illustration.Legend
        options.location string {mustBeMember(options.location, ["north", "south"])} = "north"
        options.width {mustBeNumeric} = NaN
        options.height {mustBeNumeric} = NaN
        options.unit string {mustBeMember(options.unit, ["centimeters", "inches", "points"])} = "centimeters"
        options.fontSize {mustBeNumeric} = NaN
        options.lineLength {mustBeNumeric} = NaN
        options.shrink (1,1) logical = false
        options.gap (1,1) {mustBeNumeric} = 2
    end
    fig = ancestor(lgd, "figure");

    % Resize figure to final export size
    set(fig, Units=options.unit);
    pos = get(fig, "Position");
    if isnan(options.width);  options.width = pos(3);  end
    if isnan(options.height); options.height = pos(4); end
    set(fig, Position=[pos(1), pos(2), options.width, options.height]);

    % Apply legend formatting, which determines the legend size
    if ~isnan(options.fontSize)
        set(lgd, FontSize=options.fontSize)
    end
    if ~isnan(options.lineLength)
        set(lgd, ItemTokenSize=[options.lineLength; 0])
    end
    set(lgd, Units="normalized");
    drawnow

    % Legend area height normalized to the figure, including the edge
    % padding and the gap towards the remaining content
    figPixels = getpixelposition(fig);
    lgdPixels = getpixelposition(lgd, true);
    pad = 1 / figPixels(4);
    gap = options.gap * get(groot, "ScreenPixelsPerInch") / 72 / figPixels(4);
    legendHeight = lgdPixels(4) / figPixels(4) + pad + gap;

    % Shrink remaining content (axes, tiled layouts) out of the legend area
    if options.shrink
        isNorth = options.location == "north";
        for c = fig.Children'
            if isa(c, "matlab.graphics.illustration.Legend") || ~isprop(c, "OuterPosition")
                continue
            end
            set(c, Units="normalized");
            outerPos = c.OuterPosition;
            outerPos(2) = outerPos(2) * (1 - legendHeight) + ~isNorth * legendHeight;
            outerPos(4) = outerPos(4) * (1 - legendHeight);
            set(c, OuterPosition=outerPos);
        end
        drawnow
    end

    % Compute legend position normalized to parent
    if lgd.Parent == fig
        parentPos = [0, 0, 1, 1];
    else
        set(lgd.Parent, Units="normalized");
        parentPos = lgd.Parent.OuterPosition;
    end
    toParent = @(p) [(p(1:2) - parentPos(1:2)) ./ parentPos(3:4), p(3:4) ./ parentPos(3:4)];

    % Center legend at the top/bottom figure edge
    [w, h] = deal(lgd.Position(3) * parentPos(3), lgd.Position(4) * parentPos(4));
    if options.location == "north"
        lgd.Position = toParent([(1 - w) / 2, 1 - h - pad, w, h]);
    else
        lgd.Position = toParent([(1 - w) / 2, pad, w, h]);
    end
end
