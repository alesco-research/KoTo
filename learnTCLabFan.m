%% Learn data-driven predictors for the simulated TC-Lab with fan
%  Plant (TCLabFan):     u1 drives the heater, u2 drives a cooling fan
%                        (state-input coupling). Data are simulated by
%                        generateTCLabFanData (multisine inputs, random
%                        initial temperatures in [20, 30] °C).
%  State (IODelayEmbedding):  q_t = [x_t; ...; x_{t-ny+1}; u_{t-1}; ...; u_{t-nu}]
%  Lifted state:         z_t = [1; q_t]          (liftType = "affine")
%  Predictors:           Linear    z+ = A z + B u
%                        Bilinear  z+ = A z + B u + N (u (x) z)
%                        GeKo      z+ = K (z (x) v),  v = [1; u]
%                        KCF       z+ = A z + A12 (v (x) [1; x_t])
%                        Willems  fundamental lemma, memory = ny samples
%  Embedding lengths and ridge weights were selected by trajectory-wise
%  4-fold cross-validation on the training data (see README.pdf).
clc, clearvars, close all;
initWorkspace;
%#ok<*UNRCH>

%% Set parameters
rng(42); % for reproducibility
savePlots = true;
saveModels = false;
plotTrainData = true;
plotTestData = true;
plotPrefix = "data/fan_";

% Data
numberOfTrainRuns = 14;
numberOfTestRuns = 6;
runLength = 400; % [samples]

% Embedding and lifting
numberOfOutputs = 20;  % past temperatures in the state [samples]
numberOfInputs = 20;   % past heater inputs in the state [samples]
liftType = "affine";   % "affine": z = [1; q], "rff": z = [1; q; random Fourier features]

% Ridge weights per predictor (cross-validated)
gamma = struct(Linear=1E-6, Bilinear=1E-4, GeKo=1E-4, KCF=1E-4);

% Willems fundamental lemma
flMemory = numberOfOutputs; % past samples fixing the initial condition
flHorizon = 40;             % samples predicted per block

% Errors are averaged over t >= evaluationStart (t = 0 is the first sample)
evaluationStart = 40; % [samples]

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

%% Instantiate scaling, embedding and lifting
[outputDim, inputDim] = deal(2, 3); % measured temperatures, heater and fan inputs
stateScaling = Scale([23, 70; 23, 70]); % [x1] = °C, [x2] = °C
inputScaling = Scale([0, 100; 0, 100; 0, 100]); % [u1] = %, [u2] = %, [u3] = %

embedding = IODelayEmbedding(outputDim, inputDim, numberOfOutputs, numberOfInputs);
memory = embedding.memory;
switch liftType
    case "affine"
        stateLifting = PolynomialLift(embedding.featureDim, 1);
    case "rff"
        stateLifting = RandomFourierLift(embedding.featureDim, 200, 64);
    otherwise
        error("Unknown liftType ""%s"".", liftType);
end
inputLifting = PolynomialLift(inputDim, 1); % v = [1; u]

%% Simulate data
dataDir = fullfile(fileparts(which("learnTCLab")), "data");
fprintf("\n[1] Simulate training and test trajectories...\n");
[XTrain, UTrain, XTest, UTest, Ts] = generateTCLabFanData( ...
    numberOfTrainRuns=numberOfTrainRuns, numberOfTestRuns=numberOfTestRuns, runLength=runLength);
fprintf( ...
    "\tTraining trajectories: %d (%d samples), test trajectories: %d (%d samples)\n", ...
    numel(XTrain), sum(cellfun(@(u) size(u, 2), UTrain)), ...
    numberOfTestRuns, sum(cellfun(@(u) size(u, 2), UTest)) ...
);

dataLabels = {
    ["Temp. 1", "$[^\circ$C]"];
    ["Temp. 2", "$[^\circ$C]"]
};
inputLabels = {
    ["Heater 1", "[\%]"];
    ["Heater 2", "[\%]"]
    ["Fan", "[-]"]
};

if plotTrainData
    fig = plotTrajectories( ...
        XTrain, trajectoryOptions(labels=dataLabels), ...
        UTrain, trajectoryOptions(labels=inputLabels), ...
        name="Training Data", dt=Ts);
    if savePlots
        exportPlot( ...
            fig, plotPrefix + "train_data.png", width=width, height=height, ...
            margin=margin, tickFontSize=tickFontSize, labelFontSize=labelFontSize ...
        );
    end
end
if plotTestData
    fig = plotTrajectories( ...
        XTest, trajectoryOptions(labels=dataLabels), ...
        UTest, trajectoryOptions(labels=inputLabels), ...
        name="Test Data", dt=Ts);
    if savePlots
        exportPlot( ...
            fig, plotPrefix + "test_data.png", width=width, height=height, ...
            margin=margin, tickFontSize=tickFontSize, labelFontSize=labelFontSize ...
        );
    end
end

%% Scale, embed and lift
XTrainScaled = cellfun(@stateScaling.scale, XTrain, UniformOutput=0);
UTrainScaled = cellfun(@inputScaling.scale, UTrain, UniformOutput=0);
XTestScaled = cellfun(@stateScaling.scale, XTest, UniformOutput=0);
UTestScaled = cellfun(@inputScaling.scale, UTest, UniformOutput=0);

fprintf("\n[2] Embed and lift training data...\n");
[QTrain, WTrain] = cellfun(@embedding.embed, XTrainScaled, UTrainScaled, UniformOutput=0);
[QTest, WTest] = cellfun(@embedding.embed, XTestScaled, UTestScaled, UniformOutput=0);
ZTrain = cellfun(@stateLifting.lift, QTrain, UniformOutput=0);
VTrain = cellfun(@inputLifting.lift, WTrain, UniformOutput=0);
ZTest = cellfun(@stateLifting.lift, QTest, UniformOutput=0);
VTest = cellfun(@inputLifting.lift, WTest, UniformOutput=0);

reprojectionError = stateLifting.fitProjection(cell2mat(QTrain(:)'), gamma=1E-8);
fprintf("\tState dimension: %d, lifted dimension: %d, reprojection error: %.2e\n", ...
    embedding.featureDim, stateLifting.featureDim, reprojectionError);

%% Fit predictors
fprintf("\n[3] Fit predictors...\n");
[YPairs, ZPairs, WPairs, VPairs] = getOneStepPairs(ZTrain, WTrain, VTrain, maxNumberOfPairs=1E7);
fprintf("\tNumber of one-step pairs: %d\n", size(ZPairs, 2));

% Drop trajectories that are too short for Willems fundamental lemma
isLongEnough = cellfun(@(u) size(u, 2) >= flMemory + flHorizon, UTrainScaled);
pairs = [
    cellfun(@(x) x(:, 1:end-1), XTrainScaled(isLongEnough)', UniformOutput=0);
    UTrainScaled(isLongEnough)'
];
fl = FundamentalLemma(pairs{:}, memory=flMemory, horizon=flHorizon);

predictors = {
    LinearKoopman(YPairs, ZPairs, WPairs, gamma=gamma.Linear)
    BilinearKoopman(YPairs, ZPairs, WPairs, gamma=gamma.Bilinear)
    GeKoKoopman(YPairs, ZPairs, VPairs, gamma=gamma.GeKo)
    KCFKoopman(YPairs, ZPairs, VPairs, couplingIdxs=1:1 + outputDim, gamma=gamma.KCF)
};
predictorNames = cellfun(@(c) erase(string(class(c)), "Koopman"), predictors);

usesLiftedInput = [false, false, true, true];
for k = 1:numel(predictors)
    fprintf("\t%s", evalc("predictors{" + k + "}.getOperatorNorm();"))
end

modelNames = ["Willems", predictorNames{:}];
numberOfModels = numel(modelNames);

%% Evaluate on test data
fprintf("\n[4] Evaluate prediction errors...\n");
XPredicted = cell(numberOfTestRuns, 1);
stateError = cell(numberOfTestRuns, 1);
meanError = nan(numberOfTestRuns, numberOfModels, 2);
for i = 1:numberOfTestRuns
    [x, u] = deal(XTestScaled{i}, UTestScaled{i});
    [N, T] = deal(size(x, 2), size(u, 2));
    XPredictedScaled = nan(outputDim, N, numberOfModels);
    XOneStepScaled = nan(outputDim, N, numberOfModels);

    % Evaluate Willems
    m = fl.memory;
    if T > m
        XPredictedScaled(:, m+1:T, 1) = fl.forecast(u(:, 1:m), x(:, 1:m), u(:, m+1:T));
        for t = m+1:T
            XOneStepScaled(:, t, 1) = fl.forecast(u(:, t-m:t-1), x(:, t-m:t-1), u(:, t));
        end
    end

    % Evaluate Koopman predictors
    for k = 1:numel(predictors)
        if usesLiftedInput(k)
            inputs = VTest{i};
        else
            inputs = WTest{i};
        end
        ZPredicted = predictors{k}.rollout(ZTest{i}(:, 1), inputs);
        XPredictedScaled(:, memory+1:N, k+1) = embedding.unembed(stateLifting.project(ZPredicted));
        ZOneStep = predictors{k}.predict(ZTest{i}(:, 1:end-1), inputs);
        XOneStepScaled(:, memory+2:N, k+1) = embedding.unembed(stateLifting.project(ZOneStep));
    end

    % Errors in °C
    unscaleAll = @(X) reshape(stateScaling.unscale(reshape(X, outputDim, [])), outputDim, N, numberOfModels);
    XPredicted{i} = unscaleAll(XPredictedScaled);
    XOneStep = unscaleAll(XOneStepScaled);
    stateError{i} = [
        vecnorm(XOneStep - XTest{i}, 2, 1);
        vecnorm(XPredicted{i} - XTest{i}, 2, 1)
    ];

    window = evaluationStart+1:T;
    for row = 1:2
        meanError(i, :, row) = reshape(mean(stateError{i}(row, window, :), 2, "omitnan"), 1, []);
    end
end

titles = ["one-step", "open-loop (multi-step)"];
for row = [2, 1]
    fprintf("\n\tMean %s error norm [degC], t >= %d s\n\tNr. ", titles(row), evaluationStart * Ts);
    fprintf("  %10s", modelNames);
    fprintf("\n");
    for i = 1:numberOfTestRuns
        fprintf("\t%-3d ", i);
        fprintf("  %10.3f", meanError(i, :, row));
        fprintf("\n");
    end
    fprintf("\tMean");
    fprintf("  %10.3f", mean(meanError(:, :, row), 1, "omitnan"));
    fprintf("\n");
end

%% Plot
fprintf("\n[5] Plotting...\n");
[fig, axs] = plotTrajectories( ...
    XPredicted, trajectoryOptions(labels={
        ["Temp. 1", "$[^\circ$C]"];
        ["Temp. 2", "$[^\circ$C]"]
    }), ...
    name="State Trajectories", ...
    dt=Ts ...
);
plotTrajectories(XTest, trajectoryOptions(lineStyle="--", color="k"), axs=axs, dt=Ts);
grid(axs, "on")
lgd = legend(axs(1), [modelNames(:); "True"], Orientation="horizontal");
headerLegend( ...
    lgd, ...
    width=width, ...
    height=height, ...
    fontSize=legendFontSize, ...
    lineLength=legendLineLength, ...
    shrink=true, ...
    gap=-5 ...
);
if savePlots
    exportPlot( ...
        fig, plotPrefix + "state_trajectories.pdf", ...
        width=width, height=height, margin=margin, ...
        tickFontSize=tickFontSize, labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, legendLineLength=legendLineLength ...
    );
end

[fig, axs] = plotTrajectories( ...
    stateError, trajectoryOptions(labels={
        ["One-Step", "$\left\|\hat{\bf{x}} - \bf{x}^\star\right\|$"];
        ["Multi-Step", "$\left\|\hat{\bf{x}} - \bf{x}^\star\right\|$"]
    }), ...
    name="State-space Error", ...
    dt=Ts ...
);
grid(axs, "on")
set(axs, "YScale", "log");
lgd = legend(axs(1), modelNames(:), Orientation="horizontal");
headerLegend( ...
    lgd, ...
    width=width, ...
    height=height, ...
    fontSize=legendFontSize, ...
    lineLength=legendLineLength, ...
    shrink=true, ...
    gap=-5 ...
);
if savePlots
    exportPlot( ...
        fig, plotPrefix + "state_space_error.pdf", ...
        width=width, height=height, margin=margin, ...
        tickFontSize=tickFontSize, labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, legendLineLength=legendLineLength ...
    );
end

fig = figure(Name="Phase Portrait"); hold on, grid on, colororder(colorPalette("gem12"))
for k = 1:numberOfModels
    plot(squeeze(XPredicted{1}(1, :, k)), squeeze(XPredicted{1}(2, :, k)));
end
plot(XTest{1}(1, :), XTest{1}(2, :), "k--", LineWidth=1)
plot(XTest{1}(1, 1), XTest{1}(2, 1), "ko", MarkerSize=10, MarkerFaceColor="y")
xlabel("$x_1$ $[^\circ$C]")
ylabel("$x_2$ $[^\circ$C]")
legend([modelNames(:); "True"; "$x_0$"], Location="southeast");
if savePlots
    exportPlot( ...
        fig, plotPrefix + "phase_portrait.pdf", ...
        width=width, height=height, margin=margin, ...
        tickFontSize=tickFontSize, labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, legendLineLength=legendLineLength ...
    );
end

%% Save trained models
if saveModels
    savePath = fullfile(dataDir, "TCLabModels_fan.mat");
    save(savePath, "stateScaling", "inputScaling", "embedding", ...
        "stateLifting", "inputLifting", "predictors", "fl");
    fprintf("\n\tSaved models to %s\n", savePath);
end
