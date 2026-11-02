%% Sweep input delay and training size for the TC-Lab Koopman predictors
clc, clearvars, close all;
initWorkspace;

%% Set parameters
savePlots = true;
trainData = "TCLabTrain01.mat";
testData = "TCLabTest01.mat";

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

% Base configuration (matches learnTCLab)
base = struct( ...
    stateDelay=0, ...
    inputDelay=12, ...
    stride=2, ...
    kernelType="poly", ...
    numberOfCenters=300, ...
    maxNumberOfPairs=40000, ...
    kcfCouplingRatio=2, ...
    kcfCouplingDegree=1, ...
    gamma=5E-3, ...
    flMemory=10, ...
    flHorizon=40, ...
    evaluationStart=[] ...
);

% Sweep 1: input delays (in multiples of the stride)
inputDelays = 0:20;

% Sweep 2: training sizes, as fractions of the available pairs
trainingFractions = 1:-0.1:0.4;
numberOfRepetitions = 5;  % random pair subsets averaged per size

%% Load and scale data
dataDir = fullfile(fileparts(which("sweepTCLab")), "data");
load(fullfile(dataDir, trainData), "XTrain", "UTrain", "Ts")
TsTrain = Ts;
load(fullfile(dataDir, testData), "XTest", "UTest", "Ts")
if TsTrain ~= Ts
    error("Train and test sampling time mismatch.")
end

stateScaling = Scale([23, 70; 23, 70]); % [x1] = °C, [x2] = °C
inputScaling = Scale([0, 100; 0, 100]); % [u1] = %, [u2] = %
data = {
    cellfun(@stateScaling.scale, XTrain, UniformOutput=0)
    cellfun(@inputScaling.scale, UTrain, UniformOutput=0)
    cellfun(@stateScaling.scale, XTest, UniformOutput=0)
    cellfun(@inputScaling.scale, UTest, UniformOutput=0)
};

%% Sweep 1: input delay
fprintf("\n[1] Sweep input delay...\n");

% Evaluate all delays from the longest memory on
longestMemory = max(base.stateDelay, max(inputDelays)) * base.stride;

delayErrors = cell(numel(inputDelays), 1);
for d = 1:numel(inputDelays)
    cfg = base;
    cfg.inputDelay = inputDelays(d);
    cfg.evaluationStart = longestMemory + 1;

    rng(42);
    [delayErrors{d}, predictorNames] = evaluatePredictors(cfg, data{:});
    fprintf("\tInput delay %2d: multi-step mean error norm", inputDelays(d));
    fprintf("  %.3e", delayErrors{d}(2, :));
    fprintf("\n");
end
delayErrors = cat(3, delayErrors{:});

[fig, lgd] = plotSweep(inputDelays, delayErrors, predictorNames, "Input delay", "Input Delay Sweep");
headerLegend(lgd, ...
    location="south", ...
    width=width, ...
    height=height, ...
    fontSize=legendFontSize, ...
    lineLength=legendLineLength, ...
    shrink=true ...
);
if savePlots
    exportPlot(fig, "data/input_delay_sweep.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, ...
        legendLineLength=legendLineLength ...
    );
end

%% Sweep 2: training size
fprintf("\n[2] Sweep training size...\n");
memory = max(base.stateDelay, base.inputDelay) * base.stride;
numberOfAvailablePairs = sum(cellfun(@(x) size(x, 2) - 1 - memory, data{1}));
trainingSizes = round(trainingFractions * numberOfAvailablePairs);

% Samples available to the Fundamental Lemma (inputs paired with x(:, 1:end-1))
numberOfAvailableSamples = sum(cellfun(@(u) size(u, 2), data{2}));

sizeErrors = cell(numel(trainingSizes), 1);
for s = 1:numel(trainingSizes)
    cfg = base;
    cfg.maxNumberOfPairs = trainingSizes(s);
    cfg.evaluationStart = max(memory, cfg.flMemory) + 1;

    repetitionErrors = cell(numberOfRepetitions, 1);
    for r = 1:numberOfRepetitions
        rng(r);
        repetitionErrors{r} = evaluatePredictors(cfg, data{:});
    end

    % Fundamental Lemma on truncated trajectories (deterministic, no repetitions)
    numberOfSamples = round(trainingFractions(s) * numberOfAvailableSamples);
    [flErrors, trajectoryLengths] = evaluateFundamentalLemma(cfg, numberOfSamples, data{:});

    sizeErrors{s} = [flErrors, mean(cat(3, repetitionErrors{:}), 3)];
    fprintf("\t%5d pairs (FL: %d samples, %d trajectories): multi-step mean error norm", ...
        trainingSizes(s), sum(trajectoryLengths), numel(trajectoryLengths));
    fprintf("  %.3e", sizeErrors{s}(2, :));
    fprintf("\n");
end
sizeErrors = cat(3, sizeErrors{:});

[fig, lgd] = plotSweep(trainingFractions * 100, sizeErrors, ["Willem's"; predictorNames], ...
    "Training data [\%]", "Training Size Sweep");
headerLegend(lgd, ...
    location="south", ...
    width=width, ...
    height=height, ...
    fontSize=legendFontSize, ...
    lineLength=legendLineLength, ...
    shrink=true ...
);
if savePlots
    exportPlot(fig, "data/training_size_sweep.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, ...
        legendLineLength=legendLineLength ...
    );
end

%% Local functions
function [errors, predictorNames] = evaluatePredictors(cfg, XTrain, UTrain, XTest, UTest)
    %EVALUATEPREDICTORS Fit the Koopman predictors for one configuration and
    %   return their state-space errors on the (scaled) test data.
    %
    %   errors is 4-by-P with rows
    %       [one-step mean error norm; multi-step mean error norm;
    %        one-step max. error norm; multi-step max. error norm]
    %   of the per-step Euclidean error norms, pooled over all test
    %   trajectories. Multi-step
    %   rollouts start at cfg.evaluationStart (default memory + 1) and errors
    %   are taken from the step after on, so configurations with different
    %   memories are compared on the same time window. The last time step is
    %   excluded to match evaluateFundamentalLemma, which has no input there.
    [stateDim, inputDim] = deal(size(XTrain{1}, 1), size(UTrain{1}, 1));
    stateDelayEmbedding = DelayEmbedding(stateDim, cfg.stateDelay, cfg.stride);
    inputDelayEmbedding = DelayEmbedding(inputDim, cfg.inputDelay, cfg.stride);
    memory = max(stateDelayEmbedding.memory, inputDelayEmbedding.memory);
    dqm = memory - stateDelayEmbedding.memory;
    dwm = memory - inputDelayEmbedding.memory;

    if isempty(cfg.evaluationStart)
        cfg.evaluationStart = memory + 1;
    elseif cfg.evaluationStart <= memory
        error("evaluationStart must be larger than the memory (%d).", memory);
    end

    % Delay embedding and alignment
    align = @(e, n) e(:, 1 + n:end);
    QTrain = cellfun(@(x) align(stateDelayEmbedding.embed(x), dqm), XTrain, UniformOutput=0);
    WTrain = cellfun(@(u) align(inputDelayEmbedding.embed(u), dwm), UTrain, UniformOutput=0);
    QTest = cellfun(@(x) align(stateDelayEmbedding.embed(x), dqm), XTest, UniformOutput=0);
    WTest = cellfun(@(u) align(inputDelayEmbedding.embed(u), dwm), UTest, UniformOutput=0);
    Q = cell2mat(cellfun(@(q) q(:, 1:end-1), QTrain, UniformOutput=0)');

    % Lifting
    featureDim = stateDelayEmbedding.featureDim;
    if cfg.kernelType == "poly"
        stateLifting = PolynomialLift(featureDim, 3);
    else
        [~, centers] = kmeans(Q', cfg.numberOfCenters, MaxIter=300, Replicates=3);
        switch cfg.kernelType
            case "rbf"
                stateLifting = GaussianLift(featureDim, 0.5, centers');
            case "rq"
                stateLifting = RationalQuadraticLift(featureDim, 1, 5, centers');
            case "matern"
                stateLifting = Matern52Lift(featureDim, 0.3, centers');
        end
    end
    inputLifting = PolynomialLift(inputDelayEmbedding.featureDim, 1);

    ZTrain = cellfun(@stateLifting.lift, QTrain, UniformOutput=0);
    VTrain = cellfun(@inputLifting.lift, WTrain, UniformOutput=0);
    ZTest = cellfun(@stateLifting.lift, QTest, UniformOutput=0);
    VTest = cellfun(@inputLifting.lift, WTest, UniformOutput=0);
    stateLifting.fitProjection(Q, gamma=1e-4);

    % Fit Koopman predictors
    [YPairs, ZPairs, WPairs, VPairs] = getOneStepPairs( ...
        ZTrain, WTrain, VTrain, maxNumberOfPairs=cfg.maxNumberOfPairs ...
    );

    if cfg.kernelType == "poly"
        kcfCouplingIdxs = 1:nchoosek(featureDim + cfg.kcfCouplingDegree, cfg.kcfCouplingDegree);
    else
        Z = cell2mat(cellfun(@(z) z(:, 1:end-1), ZTrain, UniformOutput=0)');
        V = cell2mat(VTrain');
        if stateLifting.includeState
            [~, ordering] = sort(var(Z(1 + stateLifting.dim:end, :), 0, 2), "descend");
            kcfOrdering = [1, 2, 3, 3 + ordering];
        else
            [~, kcfOrdering] = sort(var(Z, 0, 2), "descend");
        end
        effectiveInputs = sum(var(V, 0, 2) > 1E-10);
        numberOfCouplings = round((cfg.kcfCouplingRatio - 1) * stateLifting.featureDim / effectiveInputs);
        numberOfCouplings = clip(numberOfCouplings, 1, stateLifting.featureDim);
        kcfCouplingIdxs = sort(kcfOrdering(1:numberOfCouplings))';
    end

    predictors = {
        LinearKoopman(YPairs, ZPairs, WPairs, gamma=cfg.gamma)
        BilinearKoopman(YPairs, ZPairs, WPairs, gamma=cfg.gamma)
        GeKoKoopman(YPairs, ZPairs, VPairs, gamma=cfg.gamma)
        KCFKoopman(YPairs, ZPairs, VPairs, couplingIdxs=kcfCouplingIdxs, gamma=cfg.gamma)
    };
    testInputs = {WTest, WTest, VTest, VTest};
    predictorNames = cellfun(@(c) erase(string(class(c)), "Koopman"), predictors);

    % Evaluate on the common window
    t0 = cfg.evaluationStart;
    c0 = t0 - memory;  % column of ZTest/WTest belonging to time step t0
    [oneStepErrors, multiStepErrors] = deal(cell(numel(XTest), numel(predictors)));
    for i = 1:numel(XTest)
        XTrue = XTest{i}(:, t0+1:end-1);
        for k = 1:numel(predictors)
            w = testInputs{k}{i}(:, c0:end-1);

            ZMultiStep = predictors{k}.rollout(ZTest{i}(:, c0), w);
            XMultiStep = stateDelayEmbedding.unembed(stateLifting.project(ZMultiStep(:, 2:end)));
            multiStepErrors{i, k} = XMultiStep - XTrue;

            ZOneStep = predictors{k}.predict(ZTest{i}(:, c0:end-2), w);
            XOneStep = stateDelayEmbedding.unembed(stateLifting.project(ZOneStep));
            oneStepErrors{i, k} = XOneStep - XTrue;
        end
    end

    errors = zeros(4, numel(predictors));
    for k = 1:numel(predictors)
        errors(:, k) = errorMetrics([oneStepErrors{:, k}], [multiStepErrors{:, k}]);
    end
end

function [errors, trajectoryLengths] = evaluateFundamentalLemma(cfg, numberOfSamples, XTrain, UTrain, XTest, UTest)
    %EVALUATEFUNDAMENTALLEMMA Fit the Fundamental Lemma predictor on at most
    %   numberOfSamples training samples and return its 4-by-1 state-space
    %   errors on the same window as evaluatePredictors.
    %
    %   The training data is reduced by truncating the longest trajectories
    %   first, see truncateTrajectories.
    m = cfg.flMemory;
    t0 = cfg.evaluationStart;
    if t0 < m
        error("evaluationStart must be at least the Fundamental Lemma memory (%d).", m);
    end

    YTrain = cellfun(@(x) x(:, 1:end-1), XTrain, UniformOutput=0);
    [YTrain, UTrain] = truncateTrajectories(YTrain, UTrain, numberOfSamples, m + cfg.flHorizon);
    trajectoryLengths = cellfun(@(u) size(u, 2), UTrain);

    pairs = [YTrain(:)'; UTrain(:)'];
    predictor = FundamentalLemma(pairs{:}, memory=m, horizon=cfg.flHorizon);

    [oneStepErrors, multiStepErrors] = deal(cell(numel(XTest), 1));
    for i = 1:numel(XTest)
        % Multi-step: warm start on the m samples up to t0, predict t0+1, ..., T-1
        XMultiStep = predictor.forecast( ...
            UTest{i}(:, t0-m+1:t0), XTest{i}(:, t0-m+1:t0), UTest{i}(:, t0+1:end) ...
        );
        multiStepErrors{i} = XMultiStep - XTest{i}(:, t0+1:end-1);

        % One-step
        oneStepErrors{i} = zeros(size(multiStepErrors{i}));
        for t = t0+1:size(UTest{i}, 2)
            oneStepErrors{i}(:, t-t0) = predictor.forecast( ...
                UTest{i}(:, t-m:t-1), XTest{i}(:, t-m:t-1), UTest{i}(:, t) ...
            ) - XTest{i}(:, t);
        end
    end
    errors = errorMetrics([oneStepErrors{:}], [multiStepErrors{:}]);
end

function [Y, U] = truncateTrajectories(Y, U, numberOfSamples, minLength)
    %TRUNCATETRAJECTORIES Shorten trajectories to at most numberOfSamples
    %   samples in total by cutting all trajectories down to a common maximum
    %   length (water filling), so the longest ones are shortened first and
    %   shorter ones are kept untouched as long as possible. If that maximum
    %   would fall below minLength, the shortest trajectories are dropped
    %   until the remaining ones can keep at least minLength samples.
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

function errors = errorMetrics(oneStepErrors, multiStepErrors)
    %ERRORMETRICS Stack [one-step mean error norm; multi-step mean error
    %   norm; one-step max. error norm; multi-step max. error norm] of the
    %   given n-by-T error matrices, using the Euclidean norm per time step.
    [oneStepNorms, multiStepNorms] = deal(vecnorm(oneStepErrors), vecnorm(multiStepErrors));
    errors = [
        mean(oneStepNorms);
        mean(multiStepNorms);
        max(oneStepNorms);
        max(multiStepNorms)
    ];
end

function [fig, lgd] = plotSweep(x, errors, predictorNames, xLabel, name)
    %PLOTSWEEP Plot the 4-by-P-by-N sweep errors over x in a 2-by-2 layout:
    %   columns are one-/multi-step, rows are mean/max. error norm.
    fig = figure(Name=name);
    tiledlayout(2, 2, TileSpacing="tight", Padding="tight");
    columnTitles = ["One-Step", "Multi-Step"];
    rowLabels = {
        "$\mathrm{mean}(\left\|\hat{\bf{x}} \!-\! \bf{x}^\star\right\|)$";
        "$\max(\left\|\hat{\bf{x}} \!-\! \bf{x}^\star\right\|)$"
    };
    numberOfPredictors = size(errors, 2);
    axs = gobjects(2, 2);
    for row = 1:2
        for col = 1:2
            axs(row, col) = nexttile;
            hold on, grid on, colororder(colorPalette("gem12"))
            metric = reshape(errors(2 * (row - 1) + col, :, :), numberOfPredictors, [])';
            plot(x, metric, "-o");
            set(gca, YScale="log");
            if row == 1, title(columnTitles(col)); end
            if row == 2, xlabel(xLabel); end
            if col == 1, ylabel(rowLabels{row}); end
        end
    end
    linkaxes(axs(:, 1), "x");
    linkaxes(axs(:, 2), "x");
    set(axs(1, :), XTickLabel=[]);
    lgd = legend(axs(1, 1), predictorNames, Orientation="horizontal");
end