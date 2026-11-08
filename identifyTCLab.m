%% Identify TC-Lab models of different state dimension (not in the paper)
%  Data:          Recorded TC-Lab experiments (trainData, validationData,
%                 testData in data/)
%  Models:        NLSF   4-state physical ODE model (TCLab), parameters
%                        fitted by nonlinear least squares
%                 MOESP  linear subspace models of order d, Hankel depth
%                        selected on the validation data
%  Purpose:       If MOESP needs more states than the ODE to match the
%                 data, the measured system has dynamics beyond the ODE.
%  Output:        Single- and multi-step error table, prediction error,
%                 trajectories, MOESP singular values and order/depth
%                 sweeps (plotPrefix + *.pdf)
clc, clearvars, close all;
initWorkspace;
%#ok<*UNRCH>

%% Set parameters
rng(42); % for reproducibility
savePlots = true;
plotPrefix = "data/";

% Data
trainData = "TCLabTrain01.mat";
testData = "TCLabTest01.mat";
validationData = "TCLabTest02.mat"; % Used to select the MOESP depth

% Model parameters
moespOrders = [3, 4];   % MOESP model orders compared to the ODE
moespSweep = 1:8;       % MOESP model orders for the order sweep
moespDepths = 5:5:150;  % [s] Block-Hankel depths L swept per model order
depthCriterion = 2;     % Error minimized by the depth (1: single-step, 2: multi-step)
warmup = 10;            % Common warm-up window, >= memory of all models
parameterBounds = struct( ...
    U=[1, 15], Us=[1, 50], m=[1E-3, 1E-2], Q1max=[.5, 1.5], Q2max=[.5, 1.5] ...
);

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

stateDim = TCLab.stateDim;
outputDim = TCLab.outputDim;
stateScaling = Scale([23, 70; 23, 70]); % [x1] = °C, [x2] = °C
inputScaling = Scale([0, 100; 0, 100]); % [u1] = %, [u2] = %

dataDir = fullfile(fileparts(which("identifyTCLab")), "data");
trainPath = fullfile(dataDir, trainData);
testPath = fullfile(dataDir, testData);
validationPath = fullfile(dataDir, validationData);

%% Load data
fprintf("\n[1] Load training, validation and test trajectories...\n");
load(trainPath, "XTrain", "UTrain", "Ts"); TsTrain = Ts;
validation = load(validationPath, "XTest", "UTest", "Ts");
[XVal, UVal] = deal(validation.XTest, validation.UTest);
if TsTrain ~= validation.Ts
    error("Train and validation sampling time mismatch.")
end
load(testPath, "XTest", "UTest", "Ts")
if TsTrain ~= Ts
    error("Train and test sampling time mismatch.")
end
numberOfTestRuns = numel(XTest);
testHorizons = cellfun(@(x) size(x, 2), XTest);

fprintf( ...
    "\tTraining trajectories: %d, total number of samples: %d\n", ...
    numel(XTrain), sum(cellfun(@(u) size(u, 2), UTrain)) ...
);

% Scale trajectories
XTrainScaled = cellfun(@stateScaling.scale, XTrain, UniformOutput=0);
UTrainScaled = cellfun(@inputScaling.scale, UTrain, UniformOutput=0);
XTestScaled = cellfun(@stateScaling.scale, XTest, UniformOutput=0);
UTestScaled = cellfun(@inputScaling.scale, UTest, UniformOutput=0);
XValScaled = cellfun(@stateScaling.scale, XVal, UniformOutput=0);
UValScaled = cellfun(@inputScaling.scale, UVal, UniformOutput=0);

pairs = [
    cellfun(@(x) x(:,1:end-1), XTrainScaled(:)', UniformOutput=0);
    UTrainScaled(:)'
];

%% MOESP depth sweep
fprintf("\n[2] Sweep MOESP Hankel depth...\n");
minTrainLength = min(cellfun(@(u) size(u, 2), UTrain));
depthSamples = unique(round(moespDepths / TsTrain));
depthSamples = depthSamples(depthSamples >= 1 & depthSamples <= minTrainLength);
orders = union(moespOrders, moespSweep);
depthErrors = nan(numel(orders), numel(depthSamples));
bestDepth = nan(size(orders));
for k = 1:numel(orders)
    for l = find(depthSamples > orders(k))
        model = MOESP(pairs{:}, stateDim=orders(k), L=depthSamples(l));
        err = cell(numel(XValScaled), 1);
        for i = 1:numel(XValScaled)
            err{i} = evaluate( ...
                model, XValScaled{i}, UValScaled{i}, warmup, stateScaling, ...
                singleStep=depthCriterion == 1 ...
            );
        end
        meanError = mean([err{:}], 2, "omitnan");
        depthErrors(k, l) = meanError(depthCriterion);
    end
    [minError, best] = min(depthErrors(k, :));
    bestDepth(k) = depthSamples(best);
    fprintf("\td = %d: L = %d (%g s), validation error %.4e degC\n", ...
        orders(k), bestDepth(k), bestDepth(k) * TsTrain, minError);
end
depthOf = @(n) bestDepth(orders == n);

%% Fit models
fprintf("\n[3] Fit models...\n");
models = {};
models{end+1} = NLSF( ...
    TCLab(Ts=TsTrain), parameterBounds, pairs{:}, ...
    outputScaling=stateScaling, ...
    inputScaling=inputScaling ...
);
modelNames = "NLSF ($d=" + stateDim + "$)";

fprintf("\tNLSF parameters:\n");
for name = string(fieldnames(models{1}.parameters))'
    fprintf("\t\t%-6s = %.4g\n", name, models{1}.parameters.(name));
end

for n = moespOrders
    models{end+1} = MOESP(pairs{:}, stateDim=n, L=depthOf(n)); %#ok<SAGROW>
    modelNames(end+1) = "MOESP ($d=" + n + "$)"; %#ok<SAGROW>

    % Time constants of the identified modes
    lambda = eig(models{end}.A);
    fprintf("\tMOESP (d=%d) time constants [s]: %s\n", ...
        n, strjoin(compose("%.1f", sort(-TsTrain ./ log(abs(lambda)), "descend")), ", "));
end
numberOfModels = numel(models);

if any(cellfun(@(c) c.memory, models) > warmup)
    error("Warm-up window of %d samples is shorter than a model memory.", warmup);
end

%% Evaluate prediction error
fprintf("\n[4] Evaluate prediction errors...\n");
XPredicted = arrayfun(@(len) nan(outputDim, len, numberOfModels), testHorizons, UniformOutput=0);
stateError = arrayfun(@(len) nan(2, len, numberOfModels), testHorizons, UniformOutput=0);
for i = 1:numberOfTestRuns
    for j = 1:numberOfModels
        [stateError{i}(:, :, j), XPredictedScaled] = evaluate( ...
            models{j}, XTestScaled{i}, UTestScaled{i}, warmup, stateScaling ...
        );
        XPredicted{i}(:, warmup+1:end-1, j) = stateScaling.unscale(XPredictedScaled);
    end
end

meanStateErrors = cellfun(@(err) mean(err, 2, "omitnan"), stateError, UniformOutput=0);

% Print mean error norms
plainNames = erase(modelNames, ["$", "_", "{", "}"]);
fprintf("\tPer-trajectory single-step mean error norm [degC]\n\tNr.");
fprintf("  %16s", plainNames);
fprintf("\n");
for i = 1:numberOfTestRuns
    fprintf("\t%-3d", i);
    fprintf("  %16.4e", meanStateErrors{i}(1,:));
    fprintf("\n");
end

fprintf("\n\tPer-trajectory multi-step mean error norm [degC]\n\tNr.");
fprintf("  %16s", plainNames);
fprintf("\n");
for i = 1:numberOfTestRuns
    fprintf("\t%-3d", i);
    fprintf("  %16.4e", meanStateErrors{i}(2,:));
    fprintf("\n");
end

%% MOESP order sweep
fprintf("\n[5] Sweep MOESP model order...\n");
sweepErrors = nan(2, numel(moespSweep));
for k = 1:numel(moespSweep)
    model = MOESP(pairs{:}, stateDim=moespSweep(k), L=depthOf(moespSweep(k)));
    err = cell(numberOfTestRuns, 1);
    for i = 1:numberOfTestRuns
        err{i} = evaluate(model, XTestScaled{i}, UTestScaled{i}, warmup, stateScaling);
    end
    sweepErrors(:, k) = mean([err{:}], 2, "omitnan");
    fprintf("\td = %d: single-step %.4e, multi-step %.4e degC\n", moespSweep(k), sweepErrors(:, k));
end
nlsfErrors = mean(cell2mat(cellfun(@(err) err(:,:,1), stateError(:)', UniformOutput=0)), 2, "omitnan");

%% Plot
fprintf("\n[6] Plotting...\n");
[fig, axs] = plotTrajectories( ...
    stateError, trajectoryOptions(labels={
        ["Single-Step", "Error $[^\circ$C]"];
        ["Multi-Step", "Error $[^\circ$C]"]
    }), ...
    name="State-space Prediction Error", ...
    dt = TsTrain ...
);
grid(axs, "on")
set(axs,'YScale','log');
lgd = legend(axs(1), modelNames, Orientation="horizontal");
headerLegend(lgd, ...
    width=width, ...
    height=height, ...
    fontSize=legendFontSize, ...
    lineLength=legendLineLength, ...
    shrink=true, ...
    gap=-5 ...
);
if savePlots
    exportPlot(fig, plotPrefix + "identification_error.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, ...
        legendLineLength=legendLineLength ...
    );
end

[fig, axs] = plotTrajectories( ...
    XPredicted, trajectoryOptions(labels={
            ["Heater 1", "$[^\circ$C]"];
            ["Heater 2", "$[^\circ$C]"]
    }), ...
    name="State Trajectories", ...
    dt = TsTrain ...
);
plotTrajectories(XTest, trajectoryOptions(lineStyle="--", color="k"), axs=axs, dt=TsTrain);
grid(axs, "on")
lgd = legend(axs(1), [modelNames(:); "True"], Orientation="horizontal");
headerLegend(lgd, ...
    width=width, ...
    height=height, ...
    fontSize=legendFontSize, ...
    lineLength=legendLineLength, ...
    shrink=true, ...
    gap=-5 ...
);
if savePlots
    exportPlot(fig, plotPrefix + "identification_trajectories.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, ...
        legendLineLength=legendLineLength ...
    );
end

% Singular values of the projected output Hankel matrix indicate the order
sigma = models{2}.sigma;
numberOfSigmas = min(numel(sigma), 2 * max(moespSweep));
fig = figure(Name="MOESP Singular Values"); hold on, grid on
semilogy(1:numberOfSigmas, sigma(1:numberOfSigmas) / sigma(1), "o-", MarkerSize=4);
xline(stateDim + 0.5, "k--", HandleVisibility="off");
set(gca, YScale="log");
xlabel("Index $i$")
ylabel("$\sigma_i / \sigma_1$")
if savePlots
    exportPlot(fig, plotPrefix + "moesp_singular_values.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

titles = ["Single-Step", "Multi-Step"];
fig = figure(Name="MOESP Order Sweep");
tiledlayout(1, 2, TileSpacing="tight", Padding="tight");
for row = 1:2
    nexttile, hold on, grid on
    plot(moespSweep, sweepErrors(row, :), "o-", MarkerSize=4);
    yline(nlsfErrors(row), "k--");
    set(gca, YScale="log");
    title(titles(row))
    xlabel("Model order $d$")
    if row == 1
        ylabel("Error $[^\circ$C]")
        legend(["MOESP", modelNames(1)], Location="northeast");
    end
end
if savePlots
    exportPlot(fig, plotPrefix + "moesp_order_sweep.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, ...
        legendLineLength=legendLineLength ...
    );
end

fig = figure(Name="MOESP Depth Sweep"); hold on, grid on, colororder(colorPalette("gem12"))
plot(depthSamples * TsTrain, depthErrors', "o-", MarkerSize=2);
set(gca, ColorOrderIndex=1);
for k = 1:numel(orders)
    plot(bestDepth(k) * TsTrain, depthErrors(k, depthSamples == bestDepth(k)), ...
        "p", MarkerSize=8, HandleVisibility="off");
end
set(gca, YScale="log");
xlabel("Hankel depth $L$ [s]")
ylabel(["Multi-Step", "Error $[^\circ$C]"])
legend("$d=" + orders + "$", Location="northeast", NumColumns=2);
if savePlots
    exportPlot(fig, plotPrefix + "moesp_depth_sweep.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, ...
        legendLineLength=legendLineLength ...
    );
end

function [err, XPredictedScaled] = evaluate(model, X, U, m, scaling, options)
% EVALUATE  Single-step (row 1) and multi-step (row 2) output error norms in
% physical units (°C) of a warm-started predictor on one scaled test
% trajectory. The first m samples are used as warm-up, XPredictedScaled
% holds the scaled multi-step forecast for X(:, m+1:end-1). scaling maps
% physical to scaled outputs. Set singleStep=false to skip the single-step
% errors (NaN).
    arguments
        model (1,1)
        X (:,:)
        U (:,:)
        m (1,1) {mustBeInteger, mustBePositive}
        scaling (1,1) Scale
        options.singleStep (1,1) logical = true
    end
    err = nan(2, size(X, 2));
    errorNorm = @(XHat, XTrue) vecnorm(scaling.unscale(XHat) - scaling.unscale(XTrue), 2, 1);

    % Multi-step predictions
    XPredictedScaled = model.forecast(U(:, 1:m), X(:, 1:m), U(:, m+1:end));
    err(2, m+1:end-1) = errorNorm(XPredictedScaled, X(:, m+1:end-1));

    % Single-step predictions
    if ~options.singleStep
        return
    end
    for t = m+1:size(U, 2)
        xNextScaled = model.forecast(U(:, t-m:t-1), X(:, t-m:t-1), U(:, t));
        err(1, t) = errorNorm(xNextScaled, X(:, t));
    end
end
