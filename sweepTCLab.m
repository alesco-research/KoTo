%% Sweep training size and embedding lengths for the TC-Lab predictors
%  Data:          Recorded TC-Lab experiments (trainData, testData in data/)
%  Pipeline:      Koopman predictors as in learnTCLab (IODelayEmbedding,
%                 affine lifting, per-predictor ridge weights)
%  Sweep 1:       Training data from 10 % to 100 % of the single-step
%                 pairs at the base embedding (random subsets, averaged
%                 over numberOfRepetitions)
%  Sweep 2:       Coarse grid over numberOfOutputs and numberOfInputs,
%                 plotted as slices over numberOfOutputs for fixed
%                 numberOfInputs
%  Errors:        Mean error norms in °C over t >= evaluationStart (or
%                 t > memory for longer embeddings), averaged over the
%                 test runs
%  Output:        Training size sweep and single-/multi-step delay sweeps
%                 (plotPrefix + *.pdf)
clc, clearvars, close all;
initWorkspace;
%#ok<*UNRCH>

%% Set parameters
savePlots = true;
plotPrefix = "data/";

% Data
trainData = "TCLabTrain01.mat";
testData = "TCLabTest01.mat";

% Base configuration (matches learnTCLab)
base = struct( ...
    numberOfOutputs=20, ...
    numberOfInputs=10, ...
    liftType="affine", ...
    gamma=struct(Linear=1E-4, Bilinear=1E-4, GeKo=1E-4, KCF=1E-4), ...
    maxNumberOfPairs=1E7, ...
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

%% Sweep 1: training size
fprintf("\n[1] Sweep training size...\n");
warning("off", "MATLAB:nearlySingularMatrix");
memory = max(base.numberOfOutputs, base.numberOfInputs);
numberOfAvailablePairs = sum(cellfun(@(u) size(u, 2) - memory, data.UTrain));
trainingSizes = round(trainingFractions * numberOfAvailablePairs);

sizeErrors = cell(numel(trainingSizes), 1);
for s = 1:numel(trainingSizes)
    cfg = base;
    cfg.maxNumberOfPairs = trainingSizes(s);

    repetitionErrors = cell(numberOfRepetitions, 1);
    for r = 1:numberOfRepetitions
        rng(r);
        [repetitionErrors{r}, predictorNames] = evaluatePredictors(cfg, data);
    end

    sizeErrors{s} = mean(cat(3, repetitionErrors{:}), 3);
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

%% Sweep 2: embedding lengths
fprintf("\n[2] Sweep numberOfOutputs and numberOfInputs...\n");
[numberOfNy, numberOfNu] = deal(numel(numberOfOutputsSweep), numel(numberOfInputsSweep));
delayErrors = nan(2, numel(predictorNames), numberOfNy, numberOfNu);
for a = 1:numberOfNy
    cfg = base;
    cfg.numberOfOutputs = numberOfOutputsSweep(a);
    for b = 1:numberOfNu
        cfg.numberOfInputs = numberOfInputsSweep(b);
        rng(42);
        delayErrors(:, :, a, b) = evaluatePredictors(cfg, data);
        fprintf("\tny = %2d, nu = %2d", cfg.numberOfOutputs, cfg.numberOfInputs);
        fprintf("\n");
    end
end
warning("on", "MATLAB:nearlySingularMatrix");

titles = ["single-step", "multi-step"];
for row = [2, 1]
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

titles = ["Single-Step", "Multi-Step"];
yTicks = {[], [0.5, 1, 1.5, 2.5]};
for row = 1:2
    [fig, lgd] = plotSlices( ...
        numberOfInputsSweep, numberOfOutputsSweep, ...
        reshape(...
            delayErrors(row, :, :, :), ...
            numel(predictorNames), ...
            numberOfNy, numberOfNu ...
        ), ...
        predictorNames, titles(row), "Past Inputs:", ...
        yTicks{row} ...
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
        exportPlot(fig, plotPrefix + "delay_sweep_" + lower(erase(titles(row), "-")) + ".pdf", ...
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
    %   [single-step; multi-step], on the test data.
    %
    %   Multi-step rollouts start at sample t0 = cfg.evaluationStart (1-based
    %   index), or at the memory for longer embeddings, and are evaluated on
    %   t0+1, ..., T. As in learnTCLab, the last output sample is excluded.
    [outputDim, inputDim] = deal(size(data.XTrain{1}, 1), size(data.UTrain{1}, 1));
    embedding = IODelayEmbedding(outputDim, inputDim, cfg.numberOfOutputs, cfg.numberOfInputs);
    memory = embedding.memory;
    t0 = max(cfg.evaluationStart, memory + 1);

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

    [YPairs, ZPairs, WPairs, VPairs] = getSingleStepPairs( ...
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

            ZSingleStep = predictors{k}.predict(ZTest{i}(:, c0:T - memory - 1), w);
            XSingleStep = embedding.unembed(stateLifting.project(ZSingleStep));

            errors(:, k, i) = [
                mean(vecnorm(data.stateScaling.unscale(XSingleStep) - XTrue));
                mean(vecnorm(data.stateScaling.unscale(XMultiStep) - XTrue))
            ];
        end
    end
    errors = mean(errors, 3);
end

function [fig, lgd] = plotSweep(x, errors, modelNames, xLabel, name)
    %PLOTSWEEP Plot the 2-by-P-by-N sweep errors over x in a 1-by-2 layout
    %   (single-step, multi-step mean error norm).
    fig = figure(Name=name);
    tiledlayout(fig, 1, 2, TileSpacing="tight", Padding="tight");
    linestyleorder(fig, lineStyles("mixedstyles"))
    colororder(fig, colorPalette("gem12"))

    columnTitles = ["Single-Step", "Multi-Step"];
    numberOfModels = size(errors, 2);
    axs = gobjects(1, 2);
    for col = 1:2
        axs(col) = nexttile;
        hold on, grid on
        plot(x, reshape(errors(col, :, :), numberOfModels, [])', Marker="o");
        set(gca, YScale="log", YLimitMethod="tight");

        limits = ylim;
        ylim([limits(1), 10^round(log10(limits(1)) + 1)]);
        title(columnTitles(col));
        xlabel(xLabel);
        if col == 1
            ylabel("Error $[^\circ$C]");
        end
    end
    lgd = legend(axs(1), modelNames, Orientation="horizontal");
end

function [fig, lgd] = plotSlices(nu, ny, errors, predictorNames, name, legendLabel, yTicks)
    %PLOTSLICES Plot the P-by-numel(ny)-by-numel(nu) errors over ny in one
    %   tile per predictor, with one line per fixed nu (default color and
    %   line style order, in the order of nu) and a legend labeled by nu,
    %   preceded in-line by legendLabel.
    fig = figure(Name=name);
    tiledlayout(2, 2, TileSpacing="tight", Padding="tight");
    axs = gobjects(numel(predictorNames), 1);
    for k = 1:numel(predictorNames)
        axs(k) = nexttile; hold on, grid on
        colororder(axs(k), colorPalette("gem12"));
        h = plot(ny, reshape(errors(k, :, :), numel(ny), numel(nu)), Marker="o");
        if k == 1, sliceLines = h; end
        set(gca, YScale="log", YLimitMethod="tight");
        title(predictorNames(k));
        if k > 2, xlabel("Past Outputs [-]"); end
        if mod(k, 2) == 1
            ylabel({name; "Error $[^\circ$C]"});
        end
        if ~isempty(yTicks)
            yticks(yTicks);
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