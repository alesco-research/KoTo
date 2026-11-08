%% Learn data-driven predictors for the simulated TC-Lab with fan
%  Data:          Simulated by generateTCLabFanData from TCLab(fanSwitch="on"):
%                 u1, u2 drive the heaters, u3 drives a cooling fan that
%                 multiplies the temperature difference to ambient
%                 (state-input coupling)
%  State:         q_t = [y_t; ...; y_{t-ny}; u_{t-1}; ...; u_{t-nu}]  (IODelayEmbedding)
%  Lifted state:  z_t = [1; q_t]  (liftType = "affine")
%  Predictors:    Linear    z+ = A z + B u
%                 Bilinear  z+ = A z + B u + N (u (x) z)
%                 GeKo      z+ = K (z (x) v),  v = [1; u]
%                 KCF       z+ = A z + A12 (u (x) [1; y_t])
%                 Willems   fundamental lemma, memory = ny + 1 samples
%  Errors:        Mean error norms in °C over t >= evaluationStart,
%                 averaged over the test runs
%  Output:        Single- and multi-step error table, state trajectories,
%                 state-space error and phase portrait (plotPrefix + *.pdf)
clc, clearvars, close all;
initWorkspace;
%#ok<*UNRCH>

%% Set parameters
rng(42); % for reproducibility
savePlots = true;
saveModels = false;
plotTrainData = false;
plotTestData = false;
plotPrefix = "data/fan_";

% Data
numberOfTrainRuns = 14;
numberOfTestRuns = 6;
runLength = 400; % [samples]

% Embedding and lifting
numberOfOutputs = 20;  % past temperatures in the state [samples]
numberOfInputs = 10;   % past heater and fan inputs in the state [samples]
liftType = "affine";   % "affine": z = [1; q], "rff": z = [1; q; random Fourier features]

% Ridge weights per predictor (cross-validated)
gamma = struct(Linear=1E-6, Bilinear=1E-4, GeKo=1E-4, KCF=1E-4, Willems=0);

% Willems fundamental lemma
flMemory = numberOfOutputs + 1; % past samples fixing the initial condition
flHorizon = 40;                 % samples predicted per block

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
    ["$T_{C1}$", "$[^\circ$C]"];
    ["$T_{C2}$", "$[^\circ$C]"]
};
inputLabels = {
    ["$q_1$", "[\%]"];
    ["$q_2$", "[\%]"];
    ["$u_{\mathrm{fan}}$", "[\%]"]
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
[YPairs, ZPairs, WPairs, VPairs] = getSingleStepPairs(ZTrain, WTrain, VTrain, maxNumberOfPairs=1E7);
fprintf("\tNumber of single-step pairs: %d\n", size(ZPairs, 2));

% Drop trajectories that are too short for Willems fundamental lemma
isLongEnough = cellfun(@(u) size(u, 2) >= flMemory + flHorizon, UTrainScaled);
pairs = [
    cellfun(@(x) x(:, 1:end-1), XTrainScaled(isLongEnough)', UniformOutput=0);
    UTrainScaled(isLongEnough)'
];
fl = FundamentalLemma(pairs{:}, memory=flMemory, horizon=flHorizon, gamma=gamma.Willems);

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
    XSingleStepScaled = nan(outputDim, N, numberOfModels);

    % Evaluate Willems, warm-started on the m samples up to y_r (column
    % t0 = r+1) like the Koopman rollouts, which start from y_r
    [m, t0] = deal(fl.memory, memory + 1);
    if T > t0
        XPredictedScaled(:, t0+1:T, 1) = fl.forecast(u(:, t0-m+1:t0), x(:, t0-m+1:t0), u(:, t0+1:T));
        for t = t0+1:T
            XSingleStepScaled(:, t, 1) = fl.forecast(u(:, t-m:t-1), x(:, t-m:t-1), u(:, t));
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
        ZSingleStep = predictors{k}.predict(ZTest{i}(:, 1:end-1), inputs);
        XSingleStepScaled(:, memory+2:N, k+1) = embedding.unembed(stateLifting.project(ZSingleStep));
    end

    % Errors in °C
    unscaleAll = @(X) reshape(stateScaling.unscale(reshape(X, outputDim, [])), outputDim, N, numberOfModels);
    XPredicted{i} = unscaleAll(XPredictedScaled);
    XSingleStep = unscaleAll(XSingleStepScaled);
    stateError{i} = [
        vecnorm(XSingleStep - XTest{i}, 2, 1);
        vecnorm(XPredicted{i} - XTest{i}, 2, 1)
    ];

    window = evaluationStart+1:T;
    for row = 1:2
        meanError(i, :, row) = reshape(mean(stateError{i}(row, window, :), 2, "omitnan"), 1, []);
    end
end

titles = ["single-step", "multi-step"];
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
        ["$T_{C1}$", "$[^\circ$C]"];
        ["$T_{C2}$", "$[^\circ$C]"]
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
        ["Single-Step", "Error $[^\circ$C]"];
        ["Multi-Step", "Error $[^\circ$C]"]
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

traj = 6;
fig = figure(Name="Phase Portrait"); hold on, grid on, colororder(colorPalette("gem12"))
for k = 1:numberOfModels
    plot(squeeze(XPredicted{traj}(1, :, k)), squeeze(XPredicted{traj}(2, :, k)));
end
plot(XTest{traj}(1, :), XTest{traj}(2, :), "k--", LineWidth=1)
plot(XTest{traj}(1, 1), XTest{traj}(2, 1), "ko", MarkerSize=5, MarkerFaceColor="y")
xlabel("$T_{C1}$ $[^\circ$C]")
ylabel("$T_{C2}$ $[^\circ$C]")
legend([modelNames(:); "True"; "$y_0$"], Location="southeast");
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
