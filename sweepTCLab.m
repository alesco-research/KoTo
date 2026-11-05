%% Sweep training size and embedding lengths for the TC-Lab predictors
%  Uses the same pipeline as learnTCLab (IODelayEmbedding, affine lifting,
%  per-predictor ridge weights, Willems with memory = numberOfOutputs).
%  Sweep 1: training data from 10 % to 100 % at the base embedding.
%  Sweep 2: coarse grid over numberOfOutputs and numberOfInputs, plotted as
%           slices over numberOfOutputs for fixed numberOfInputs.
%  All errors are mean error norms in °C, averaged over the test window
%  t >= evaluationStart (t = 0 is the first sample) and the test runs.
clc, clearvars, close all;
initWorkspace;
%#ok<*UNRCH>

%% Set parameters
savePlots = true;
plotPrefix = "data/";
trainData = "TCLabTrain01.mat";
testData = "TCLabTest01.mat";

% Base configuration (matches learnTCLab)
base = struct( ...
    numberOfOutputs=20, ...
    numberOfInputs=10, ...
    liftType="affine", ...
    gamma=struct(Linear=1E-6, Bilinear=1E-4, GeKo=1E-4, KCF=1E-4), ...
    maxNumberOfPairs=1E7, ...
    flHorizon=40, ...
    evaluationStart=40 ...
);

% Sweep 1: training sizes, as fractions of the available data
trainingFractions = 0.1:0.1:1;
numberOfRepetitions = 5;  % random pair subsets averaged per size

% Sweep 2: embedding lengths (coarse grid)
numberOfOutputsSweep = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45];
numberOfInputsSweep = [0, 5, 10, 15];

% Plotting parameters
margin = 0;
width = 8.6; % [cm]
height = 5; % [cm]
tickFontSize = 8;
labelFontSize = 10;
legendFontSize = 7;
legendLineLength = 10;
set(groot, defaultLineLineWidth=1);
set(groot, defaultLineMarkerSize=2);
set(groot, defaultTextInterpreter="latex");
set(groot, defaultLegendInterpreter="latex");
set(groot, defaultAxesTickLabelInterpreter="latex");

%% Load and scale data
dataDir = fullfile(fileparts(which("sweepTCLab")), "data");
load(fullfile(dataDir, trainData), "XTrain", "UTrain", "Ts"); TsTrain = Ts;
load(fullfile(dataDir, testData), "XTest", "UTest", "Ts")
if TsTrain ~= Ts
    error("Train and test sampling time mismatch.")
end

stateScaling = Scale([23, 70; 23, 70]); % [x1] = °C, [x2] = °C
inputScaling = Scale([0, 100; 0, 100]); % [u1] = %, [u2] = %
data = struct( ...
    XTrain={cellfun(@stateScaling.scale, XTrain(:), UniformOutput=0)}, ...
    UTrain={cellfun(@inputScaling.scale, UTrain(:), UniformOutput=0)}, ...
    XTest={cellfun(@stateScaling.scale, XTest(:), UniformOutput=0)}, ...
    UTest={cellfun(@inputScaling.scale, UTest(:), UniformOutput=0)}, ...
    XTestTrue={XTest(:)}, ...
    stateScaling=stateScaling ...
);

% Common evaluation window for all configurations of both sweeps
longestMemory = max( ...
    max(numberOfOutputsSweep) - 1, max([numberOfInputsSweep, base.numberOfInputs]));
base.evaluationStart = max([ ...
    base.evaluationStart, longestMemory + 1, max(numberOfOutputsSweep), base.numberOfOutputs]);
fprintf("\tEvaluation window: t >= %d samples\n", base.evaluationStart);

%% Sweep 1: training size
fprintf("\n[1] Sweep training size...\n");
warning("off", "MATLAB:nearlySingularMatrix");
warning("off", "FundamentalLemma:NotPersistentlyExciting");
memory = max(base.numberOfOutputs - 1, base.numberOfInputs);
numberOfAvailablePairs = sum(cellfun(@(u) size(u, 2) - memory, data.UTrain));
trainingSizes = round(trainingFractions * numberOfAvailablePairs);

% Samples available to Willems (inputs paired with x(:, 1:end-1))
numberOfAvailableSamples = sum(cellfun(@(u) size(u, 2), data.UTrain));

sizeErrors = cell(numel(trainingSizes), 1);
for s = 1:numel(trainingSizes)
    cfg = base;
    cfg.maxNumberOfPairs = trainingSizes(s);

    repetitionErrors = cell(numberOfRepetitions, 1);
    for r = 1:numberOfRepetitions
        rng(r);
        [repetitionErrors{r}, predictorNames] = evaluatePredictors(cfg, data);
    end

    sizeErrors{s} = [mean(cat(3, repetitionErrors{:}), 3)];
    fprintf( ...
        "\t%3d%% %6d pairs", round(100 * trainingFractions(s)), trainingSizes(s) ...
    )
    fprintf("\n");
end

[fig, lgd] = plotSweep( ...
    trainingFractions * 100, cat(3, sizeErrors{:}), predictorNames, ...
    "Training data [\%]", "Training Size Sweep" ...
);
headerLegend(lgd, ...
    location="south", ...
    width=width, ...
    height=height, ...
    fontSize=legendFontSize, ...
    lineLength=legendLineLength, ...
    shrink=true ...
);
if savePlots
    exportPlot(fig, plotPrefix + "training_size_sweep.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, ...
        legendLineLength=legendLineLength ...
    );
end
warning("on", "MATLAB:nearlySingularMatrix");
warning("on", "FundamentalLemma:NotPersistentlyExciting");

%% Sweep 2: embedding lengths
fprintf("\n[2] Sweep numberOfOutputs and numberOfInputs...\n");
[numberOfNy, numberOfNu] = deal(numel(numberOfOutputsSweep), numel(numberOfInputsSweep));
delayErrors = nan(2, numel(predictorNames), numberOfNy, numberOfNu);
flDelayErrors = nan(2, numberOfNy);
for a = 1:numberOfNy
    cfg = base;
    cfg.numberOfOutputs = numberOfOutputsSweep(a);

    % Willems only depends on numberOfOutputs (its memory)
    flDelayErrors(:, a) = evaluateFundamentalLemma(cfg, inf, data);

    for b = 1:numberOfNu
        cfg.numberOfInputs = numberOfInputsSweep(b);
        rng(42);
        delayErrors(:, :, a, b) = evaluatePredictors(cfg, data);
        fprintf("\tny = %2d, nu = %2d", cfg.numberOfOutputs, cfg.numberOfInputs);
        fprintf("\n");
    end
end

titles = ["one-step", "open-loop (multi-step)"];
for row = [2, 1]
    fprintf("\n\tMean %s error norm [degC], Willems over numberOfOutputs\n\tny  ", titles(row));
    fprintf("  %6d", numberOfOutputsSweep);
    fprintf("\n\t    ");
    fprintf("  %6.3f", flDelayErrors(row, :));
    fprintf("\n");
    for k = 1:numel(predictorNames)
        fprintf("\n\tMean %s error norm [degC], %s (rows: ny, columns: nu)\n\tny\\nu", ...
            titles(row), predictorNames(k));
        fprintf("  %6d", numberOfInputsSweep);
        fprintf("\n");
        for a = 1:numberOfNy
            fprintf("\t%-4d ", numberOfOutputsSweep(a));
            fprintf("  %6.3f", squeeze(delayErrors(row, k, a, :)));
            fprintf("\n");
        end
        [minError, idx] = min(delayErrors(row, k, :, :), [], "all");
        [a, b] = ind2sub([numberOfNy, numberOfNu], idx);
        fprintf("\tBest: ny = %d, nu = %d (%.3f)\n", ...
            numberOfOutputsSweep(a), numberOfInputsSweep(b), minError);
    end
end

surfaceNames = ["One-Step", "Multi-Step"];
ylimits = {[], [0.5, 1, 1.5, 2.5]};
for row = 1:2
    [fig, lgd] = plotSlices( ...
        numberOfInputsSweep, numberOfOutputsSweep, ...
        reshape(delayErrors(row, :, :, :), numel(predictorNames), ...
        numberOfNy, numberOfNu ...
        ), ...
        predictorNames, surfaceNames(row) + " Delay Sweep", "Past Inputs:", ...
        ylimits{row} ...
    );
    headerLegend(lgd, ...
        location="south", ...
        width=width, ...
        height=height, ...
        fontSize=legendFontSize, ...
        lineLength=legendLineLength, ...
        shrink=true ...
    );
    if savePlots
        exportPlot(fig, plotPrefix + "delay_sweep_" + lower(erase(surfaceNames(row), "-")) + ".pdf", ...
            width=width, ...
            height=height, ...
            margin=margin, ...
            tickFontSize=tickFontSize, ...
            labelFontSize=labelFontSize, ...
            legendFontSize=legendFontSize, ...
            legendLineLength=legendLineLength ...
        );
    end
end

%% Local functions
function [errors, predictorNames] = evaluatePredictors(cfg, data)
    %EVALUATEPREDICTORS Fit the Koopman predictors of learnTCLab for one
    %   configuration and return their 2-by-P mean error norms in °C,
    %   [one-step; multi-step], on the test data.
    %
    %   Multi-step rollouts start at sample t0 = cfg.evaluationStart (1-based
    %   index) from the embedded test state, so configurations with different
    %   memories are compared on the same window t0+1, ..., T as in
    %   learnTCLab. The last output sample is excluded to match
    %   evaluateFundamentalLemma, which has no input there.
    [outputDim, inputDim] = deal(size(data.XTrain{1}, 1), size(data.UTrain{1}, 1));
    embedding = IODelayEmbedding(outputDim, inputDim, cfg.numberOfOutputs, cfg.numberOfInputs);
    memory = embedding.memory;
    t0 = cfg.evaluationStart;
    if t0 <= memory
        error("evaluationStart must be larger than the memory (%d).", memory);
    end

    switch cfg.liftType
        case "affine"
            stateLifting = PolynomialLift(embedding.featureDim, 1);
        case "rff"
            stateLifting = RandomFourierLift(embedding.featureDim, 200, 64);
        otherwise
            error("Unknown liftType ""%s"".", cfg.liftType);
    end
    inputLifting = PolynomialLift(inputDim, 1); % v = [1; u]

    [QTrain, WTrain] = cellfun(@embedding.embed, data.XTrain, data.UTrain, UniformOutput=0);
    [QTest, WTest] = cellfun(@embedding.embed, data.XTest, data.UTest, UniformOutput=0);
    ZTrain = cellfun(@stateLifting.lift, QTrain, UniformOutput=0);
    VTrain = cellfun(@inputLifting.lift, WTrain, UniformOutput=0);
    ZTest = cellfun(@stateLifting.lift, QTest, UniformOutput=0);
    VTest = cellfun(@inputLifting.lift, WTest, UniformOutput=0);
    stateLifting.fitProjection(cell2mat(QTrain(:)'), gamma=1E-8);

    [YPairs, ZPairs, WPairs, VPairs] = getOneStepPairs( ...
        ZTrain, WTrain, VTrain, maxNumberOfPairs=cfg.maxNumberOfPairs ...
    );
    predictors = {
        LinearKoopman(YPairs, ZPairs, WPairs, gamma=cfg.gamma.Linear)
        BilinearKoopman(YPairs, ZPairs, WPairs, gamma=cfg.gamma.Bilinear)
        GeKoKoopman(YPairs, ZPairs, VPairs, gamma=cfg.gamma.GeKo)
        KCFKoopman(YPairs, ZPairs, VPairs, couplingIdxs=1:1 + outputDim, gamma=cfg.gamma.KCF)
    };
    testInputs = {WTest, WTest, VTest, VTest};
    predictorNames = cellfun(@(c) erase(string(class(c)), "Koopman"), predictors);

    % Column c of ZTest/WTest belongs to sample memory + c
    c0 = t0 - memory;
    errors = zeros(2, numel(predictors), numel(data.XTest));
    for i = 1:numel(data.XTest)
        T = size(data.UTest{i}, 2);
        XTrue = data.XTestTrue{i}(:, t0+1:T);
        for k = 1:numel(predictors)
            w = testInputs{k}{i}(:, c0:T - memory - 1);

            ZMultiStep = predictors{k}.rollout(ZTest{i}(:, c0), w);
            XMultiStep = embedding.unembed(stateLifting.project(ZMultiStep(:, 2:end)));

            ZOneStep = predictors{k}.predict(ZTest{i}(:, c0:T - memory - 1), w);
            XOneStep = embedding.unembed(stateLifting.project(ZOneStep));

            errors(:, k, i) = [
                mean(vecnorm(data.stateScaling.unscale(XOneStep) - XTrue));
                mean(vecnorm(data.stateScaling.unscale(XMultiStep) - XTrue))
            ];
        end
    end
    errors = mean(errors, 3);
end

function [errors, trajectoryLengths] = evaluateFundamentalLemma(cfg, numberOfSamples, data)
    %EVALUATEFUNDAMENTALLEMMA Fit Willems fundamental lemma with memory
    %   cfg.numberOfOutputs on at most numberOfSamples training samples and
    %   return its 2-by-1 mean error norms in °C, [one-step; multi-step], on
    %   the same window as evaluatePredictors.
    %
    %   The training data is reduced by truncating the longest trajectories
    %   first, see truncateTrajectories. Trajectories shorter than
    %   memory + horizon are dropped.
    [m, h] = deal(cfg.numberOfOutputs, cfg.flHorizon);
    t0 = cfg.evaluationStart;
    if t0 < m
        error("evaluationStart must be at least the Willems memory (%d).", m);
    end

    YTrain = cellfun(@(x) x(:, 1:end-1), data.XTrain, UniformOutput=0);
    [YTrain, UTrain] = truncateTrajectories(YTrain, data.UTrain, numberOfSamples, m + h);
    trajectoryLengths = cellfun(@(u) size(u, 2), UTrain);

    pairs = [YTrain(:)'; UTrain(:)'];
    fl = FundamentalLemma(pairs{:}, memory=m, horizon=h);

    [x, u] = deal(data.XTest, data.UTest);
    errors = zeros(2, numel(x));
    for i = 1:numel(x)
        T = size(u{i}, 2);
        XTrue = data.XTestTrue{i}(:, t0+1:T);

        % Multi-step: warm start on the m samples up to t0, predict t0+1, ..., T
        XMultiStep = fl.forecast(u{i}(:, t0-m+1:t0), x{i}(:, t0-m+1:t0), u{i}(:, t0+1:T));

        % One-step
        XOneStep = zeros(size(XTrue));
        for t = t0+1:T
            XOneStep(:, t-t0) = fl.forecast(u{i}(:, t-m:t-1), x{i}(:, t-m:t-1), u{i}(:, t));
        end

        errors(:, i) = [
            mean(vecnorm(data.stateScaling.unscale(XOneStep) - XTrue));
            mean(vecnorm(data.stateScaling.unscale(XMultiStep) - XTrue))
        ];
    end
    errors = mean(errors, 2);
end

function [Y, U] = truncateTrajectories(Y, U, numberOfSamples, minLength)
    %TRUNCATETRAJECTORIES Shorten trajectories to at most numberOfSamples
    %   samples in total by cutting all trajectories down to a common maximum
    %   length (water filling), so the longest ones are shortened first and
    %   shorter ones are kept untouched as long as possible. If that maximum
    %   would fall below minLength, the shortest trajectories are dropped
    %   until the remaining ones can keep at least minLength samples.
    %   Trajectories shorter than minLength are always dropped.
    lengths = cellfun(@(u) size(u, 2), U);
    valid = find(lengths >= minLength);
    [~, order] = sort(lengths(valid), "descend");
    order = valid(order);

    [keep, cap] = deal([]);
    for n = numel(order):-1:1
        keep = order(1:n);
        capLevels = max(lengths(keep)):-1:minLength;
        totals = arrayfun(@(c) sum(min(lengths(keep), c)), capLevels);
        cap = capLevels(find(totals <= numberOfSamples, 1));
        if ~isempty(cap)
            break
        end
    end
    if isempty(cap)
        error("Not enough samples for a single trajectory of length %d.", minLength);
    end

    keep = sort(keep);
    Y = cellfun(@(y) y(:, 1:min(end, cap)), Y(keep), UniformOutput=0);
    U = cellfun(@(u) u(:, 1:min(end, cap)), U(keep), UniformOutput=0);
end

function [fig, lgd] = plotSweep(x, errors, modelNames, xLabel, name)
    %PLOTSWEEP Plot the 2-by-P-by-N sweep errors over x in a 1-by-2 layout
    %   (one-step, multi-step mean error norm).
    fig = figure(Name=name);
    tiledlayout(1, 2, TileSpacing="tight", Padding="tight");
    columnTitles = ["One-Step", "Multi-Step"];
    numberOfModels = size(errors, 2);
    axs = gobjects(1, 2);
    for col = 1:2
        axs(col) = nexttile;
        hold on, grid on, colororder(colorPalette("gem12"))
        plot(x, reshape(errors(col, :, :), numberOfModels, [])', "-o");
        set(gca, YScale="log", YLimitMethod="tight");

        limits = ylim;
        ylim([limits(1), 10^round(log10(limits(1)) + 1)]);
        title(columnTitles(col));
        xlabel(xLabel);
        if col == 1
            ylabel( ...
                "$\mathrm{mean}(\|\hat{\mathrm{y}} - \mathrm{y}^\star\|)$ " + ...
                "[$^\circ$C]" ...
            );
        end
    end
    lgd = legend(axs(1), modelNames, Orientation="horizontal");
end

function [fig, lgd] = plotSlices(nu, ny, errors, predictorNames, name, legendLabel, ylimits)
    %PLOTSLICES Plot the P-by-numel(ny)-by-numel(nu) errors over ny in one
    %   tile per predictor, with one line per fixed nu (default color and
    %   line style order, in the order of nu) and a legend labeled by nu,
    %   preceded in-line by legendLabel.
    fig = figure(Name=name);
    tiledlayout(2, 2, TileSpacing="tight", Padding="tight");
    axs = gobjects(numel(predictorNames), 1);
    for k = 1:numel(predictorNames)
        axs(k) = nexttile; hold on, grid on
        linestyleorder(axs(k), "-");
        colororder(axs(k), colorPalette("gem12"));
        h = plot(ny, reshape(errors(k, :, :), numel(ny), numel(nu)), Marker="o");
        if k == 1, sliceLines = h; end
        set(gca, YScale="log", YLimitMethod="tight");
        title(predictorNames(k));
        if k > 2, xlabel("Past Outputs [-]"); end
        if mod(k, 2) == 1
            ylabel({
                "$\mathrm{mean}(\!\|\hat{\mathrm{y}}$-$\mathrm{y}^{\star\!}\|\!)$";
                "[$^\circ$C]"
            });
        end
        if ~isempty(ylimits)
            yticks(ylimits);
        end
    end
    linkaxes(axs);
    set(axs(1:2), XTickLabel=[]);
    set(axs([2,4]), YTickLabel=[]);
    % Invisible first entry acts as an in-line legend title
    labelLine = plot(axs(1), NaN, NaN, LineStyle="none", Marker="none");
    lgd = legend(axs(1), [labelLine; sliceLines], [legendLabel; string(nu(:))], ...
        Orientation="horizontal");
end
