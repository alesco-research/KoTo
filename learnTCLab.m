%% Learn Koopman Operator for a TC-Lab
clc, clearvars, close all;
initWorkspace;
%#ok<*UNRCH>

%% Set parameters
rng(42); % for reproducibility
savePlots = false;
saveModels = false; 
plotTrainData = true;
plotTestData = true;
trainData = "TCLabTrain1.mat";
testData = "TCLabTest1.mat";

%% Instantiate model, scaling, delay embedding and lifting
tcLab = TCLab(measurementNoiseCov=5E-2*[1, 0.5; 0.5, 1]);
[stateDim, inputDim] = deal(tcLab.stateDim, tcLab.inputDim);
stateScaling = Scale([23, 70; 23, 70]); % [x1] = K, [x2] = K
inputScaling = Scale([0, 100; 0, 100]); % [u1] = %, [u2] = %

[stateDelay, inputDelay] = deal(2, 0);
stateDelayEmbedding = DelayEmbedding(stateDim, stateDelay);
inputDelayEmbedding = DelayEmbedding(inputDim, inputDelay);
memory = max(stateDelayEmbedding.memory, inputDelayEmbedding.memory);
dqm = memory - stateDelayEmbedding.memory;
dwm = memory - inputDelayEmbedding.memory;

kernelType = "poly";
switch kernelType
    case "rbf"
        stateLifting = @(centers) GaussianLift(stateDelayEmbedding.featureDim, 0.5, centers);
        numberOfCenters = 300;
    case "rq"
        stateLifting = @(centers) RationalQuadraticLift(stateDelayEmbedding.featureDim, 1, 5, centers);
        numberOfCenters = 300;
    case "matern"
        stateLifting = @(centers) Matern52Lift(stateDelayEmbedding.featureDim, 1, centers);
        numberOfCenters = 300;
    case "poly"
        stateLifting = @(~) PolynomialLift(stateDelayEmbedding.featureDim, 3);
end
inputLifting = PolynomialLift(inputDelayEmbedding.featureDim, 5);

switch "multi"
    case "one"
        getPairs = @(Z, U, V) getOneStepPairs(Z, U, V, maxNumberOfPairs=5000);
    case "multi"
        getPairs = @(Z, U, V) getMultiStepWindows(Z, U, V, horizon=100, maxNumberOfWindows=200);
end

% KCF coupling parameters
kcfCouplingRatio = 2;   % > 1
kcfCouplingDegree = 2;  % <= polynomial lift degree

dataDir = fullfile(fileparts(which("learnTCLab")), "data");
trainPath = fullfile(dataDir, trainData);
testPath = fullfile(dataDir, testData);
savePath = fullfile(dataDir, "TCLab" + upper(kernelType) + ".mat");

%% Load/generate test data
if isfile(trainPath)
    fprintf("\n[1] Load training trajectories...\n");
    load(trainPath, "XTrain", "UTrain", "Ts")
    numberOfTrajectorySamples = numel(XTrain);
    TsTrain = Ts;
else
    fprintf("\n[1] Generate training trajectories...\n");
    TsTrain = tcLab.Ts;
    trainHorizon = 400;
    numberOfTrajectorySamples = 5;

    xRange = [23, 70; 23, 70]; % physical operating ranges
    trainTime  = (0:trainHorizon-1)' * TsTrain; % time vector
    trainFreqs  = [150, 350, 55, 95, 155, 265]; % driving input frequencies

    XTrain = cell(numberOfTrajectorySamples, 1);
    UTrain = cell(numberOfTrajectorySamples, 1);
    for l = 1:numberOfTrajectorySamples
        x0 = xRange(:,1) + diff(xRange, 1, 2) .* rand(2,1);
        uTrain = [
            1 + sum(sin(trainFreqs .* trainTime + 2 * pi * rand(1,6)), 2)';
            1 + sum(sin(trainFreqs .* trainTime + 2 * pi * rand(1,6)), 2)'
        ] * 25;
        UTrain{l} = clip(uTrain, 0, 100);
        [~, XTrain{l}] = tcLab.simulate(x0, UTrain{l});
    end
end

fprintf( ...
    "\tNumber of trajectories: %d, total number of samples: %d\n", ...
    numberOfTrajectorySamples, ...
    sum(cellfun(@(u) size(u, 2), UTrain)) ...
);

if plotTrainData
    plotTrajectories( ...
        XTrain, UTrain, name="Train Trajectories", ...
        stateLabels={
            ["Temp. Heater 1", "$[^\circ$C]"];
            ["Temp. Heater 2", "$[^\circ$C]"]
        }, ...
        inputLabels={
            ["Input Heater 1",  "[\%]"];
            ["Input Heater 2",  "[\%]"]
        } ...
    );
end

%% Scaling
XTrainScaled = cellfun(@stateScaling.scale, XTrain, UniformOutput=false);
UTrainScaled = cellfun(@inputScaling.scale, UTrain, UniformOutput=false);

% Stack trajectories
X = cell2mat(cellfun(@(x) x(:, 1:end-1), XTrainScaled, UniformOutput=false)');
U = cell2mat(UTrainScaled');

%% Delay embedding
QTrain = cellfun(@stateDelayEmbedding.embed, XTrainScaled, UniformOutput=false);
WTrain = cellfun(@inputDelayEmbedding.embed, UTrainScaled, UniformOutput=false);

% Align embeddings
QTrain = cellfun(@(e) e(:, 1 + dqm:end), QTrain, UniformOutput=false);
WTrain = cellfun(@(e) e(:, 1 + dwm:end), WTrain, UniformOutput=false);

% Stack trajectories
Q = cell2mat(cellfun(@(q) q(:, 1:end-1), QTrain, UniformOutput=false)');
W = cell2mat(WTrain');

%% Lifting
fprintf("\n[2] Lift training data...\n");
centers = [];
if any(kernelType == ["rbf", "rq", "matern"])
    [~, centers] = kmeans(Q', numberOfCenters, MaxIter=300, Replicates=3);
end
stateLifting = stateLifting(centers');

% Lift trajectories
ZTrain = cellfun(@stateLifting.lift, QTrain, UniformOutput=false);
VTrain = cellfun(@inputLifting.lift, WTrain, UniformOutput=false);

% Stack trajectories
Z = cell2mat(cellfun(@(z) z(:, 1:end-1), ZTrain, UniformOutput=false)');
V = cell2mat(VTrain');

reprojectionError = stateLifting.fitProjection(Q, gamma=1e-4);
fprintf("\tReprojection Error: %.4f\n", reprojectionError);

%% Fit Koopman controllers
fprintf("\n[3] Fit Koopman controllers...\n");
[YPairs, ZPairs, WPairs, VPairs] = getPairs(ZTrain, WTrain, VTrain);
pairs = [
    cellfun(@(x) x(:,1:end-1), XTrainScaled(:)', UniformOutput=false);
    UTrainScaled(:)'
];

memoryPredictors = {};
memoryPredictors{end+1} = MOESP(pairs{:}, L=100, stateDim = 4);
memoryPredictorNames = cellfun(@(c) string(class(c)), memoryPredictors);

predictors = {};
predictors{end+1} = LinearKoopman(YPairs, ZPairs, WPairs, gamma=5E-4);
predictors{end+1} = BilinearKoopman(YPairs, ZPairs, WPairs, gamma=5E-4);
predictors{end+1} = GeKoKoopman(YPairs, ZPairs, VPairs, gamma=5E-4);

if kernelType == "poly"
    % Get polynomial coupling
    kcfCouplingIdxs = 1:nchoosek(stateDim + kcfCouplingDegree, kcfCouplingDegree);
else
    % Get variance sorted coupling from coupling ratio
    if stateLifting.includeState
        [~, ordering] = sort(var(Z(1 + stateLifting.dim:end, :), 0, 2), "descend");
        kcfOrdering = [1, 2, 3, 3 + ordering];
    else
        [~, kcfOrdering] = sort(var(Z, 0, 2), "descend");
    end
    effectiveInputs = sum(var(V, 0, 2) > 1E-10);
    numberOfCouplings = round((kcfCouplingRatio - 1) * stateLifting.featureDim / effectiveInputs);
    numberOfCouplings = clip(numberOfCouplings, 1, stateLifting.featureDim);
    kcfCouplingIdxs = sort(kcfOrdering(1:numberOfCouplings))';
end

predictors{end+1} = KCFKoopman(YPairs, ZPairs, VPairs, couplingIdxs=kcfCouplingIdxs, gamma=5E-4);
predictorNames = cellfun(@(c) erase(string(class(c)), "Koopman"), predictors);

%% Evaluate operator norms
for i = 1:numel(predictors)
    fprintf("\t%s", evalc("predictors{" + i + "}.getOperatorNorm();"))
end

%% Load/generate test data
if isfile(testPath)
    fprintf("\n[4] Load test data...\n");
    load(testPath, "XTest", "UTest", "Ts")
    testHorizon = size(UTest{1}, 2);
    numberOfTestRuns = numel(XTest);

    testTime = (0:testHorizon) * Ts;

    if TsTrain ~= Ts
        error("Train and test sampling time mismatch.")
    end
else
    fprintf("\n[4] Generate testing trajectories...\n");
    testHorizon = 400;
    numberOfTestRuns = 3;

    testTime = (0:testHorizon) * tcLab.Ts;
    testFreqs = [100, 125, 257, 90];

    XTest = cell(numberOfTestRuns, 1);
    UTest = cell(numberOfTestRuns, 1);
    for i = 1:numberOfTestRuns
        x0 = xRange(:,1) + diff(xRange, 1, 2) .* rand(2,1);
        uTest = [
            1 + sum(sin(testFreqs .* testTime(1:end - 1)' + 2 * pi * rand(1,4)), 2)';
            1 + sum(sin(testFreqs .* testTime(1:end - 1)' + 2 * pi * rand(1,4)), 2)'
        ] * 25;
        UTest{i} = clip(uTest, 0, 100);
        XTest{i} = tcLab.simulate(x0, UTest{i});
    end
end

% Scale trajectories
XTestScaled = cellfun(@stateScaling.scale, XTest, UniformOutput=false)';
UTestScaled = cellfun(@inputScaling.scale, UTest, UniformOutput=false)';

% Delay embedding
QTest = cellfun(@stateDelayEmbedding.embed, XTestScaled, UniformOutput=false);
WTest = cellfun(@inputDelayEmbedding.embed, UTestScaled, UniformOutput=false);

% Align embeddings
QTest = cellfun(@(e) e(:, 1 + dqm:end), QTest, UniformOutput=false);
WTest = cellfun(@(e) e(:, 1 + dwm:end), WTest, UniformOutput=false);

% Lift trajectories
ZTest = cellfun(@stateLifting.lift, QTest, UniformOutput=false);
VTest = cellfun(@inputLifting.lift, WTest, UniformOutput=false);

if plotTestData
    plotTrajectories( ...
        XTest, UTest, name="Test Trajectories", ...
        stateLabels={
            ["Temp. Heater 1", "$[^\circ$C]"];
            ["Temp. Heater 2", "$[^\circ$C]"]
        }, ...
        inputLabels={
            ["Input Heater 1",  "[\%]"];
            ["Input Heater 2",  "[\%]"]
        } ...
    );
end

%% Evaluate prediction error
fprintf("\n[5] Evaluate prediction errors...\n");
numberOfPredictors = numel(memoryPredictors) + numel(predictors);

testInputs = {WTest, WTest, VTest, VTest};
XPredicted = nan(numberOfTestRuns, stateDim, testHorizon + 1, numberOfPredictors);
liftedStateError = nan(numberOfTestRuns, testHorizon + 1, numel(predictors));
scaledStateError = nan(numberOfTestRuns, testHorizon + 1, numberOfPredictors);
for i = 1:numberOfTestRuns
    % State-full predictors
    for j = 1:numel(memoryPredictors)
        m = memoryPredictors{j}.memory;
        uWarm = UTestScaled{i}(:, 1:m);
        yWarm = XTestScaled{i}(:, 1:m);
        uFuture = UTestScaled{i}(:, m+1:end);
        
        XPredictedScaled = memoryPredictors{j}.forecast(uWarm, yWarm, uFuture);

        XPredicted(i, :, m+1:end-1, j) = stateScaling.unscale(XPredictedScaled);
        scaledStateError(i, m+1:end-1, j) = sqrt( ...
            sum((XPredictedScaled - XTestScaled{i}(:, m+1:end-1)).^2, 1, "omitnan") ...
        );
    end
    % Regular predictors
    for k = 1:numel(predictors)
        ZPredicted = predictors{k}.rollout(ZTest{i}(:, 1), testInputs{k}{i});
        
        QPredicted = stateLifting.project(ZPredicted);
        XPredictedScaled = stateDelayEmbedding.unembed(QPredicted);
        XPredicted(i, :, memory+1:end, k+j) = stateScaling.unscale(XPredictedScaled);

        liftedStateError(i, memory+1:end, k) = sqrt(sum((ZPredicted - ZTest{i}).^2, 1));
        scaledStateError(i, memory+1:end, k+j) = sqrt( ...
            sum((XPredictedScaled - XTestScaled{i}(:, memory+1:end)).^2, 1) ...
        );
    end
end

meanLiftedStateErrors = mean(liftedStateError, 2, "omitnan");
meanScaledStateErrors = mean(scaledStateError, 2, "omitnan");

fprintf("\tPer-trajectory mean error (scaled state)\n\tNr.");
fprintf("  %12s", [memoryPredictorNames(:); predictorNames(:)]);
fprintf("\n");
for i = 1:numberOfTestRuns
    fprintf("\t%-3d", i);
    fprintf("  %12.4e", meanScaledStateErrors(i, :));
    fprintf("\n");
end

%% Plot
fprintf("\n[6] Plotting...\n");
margin = 0;
width = 12;
height = 8;
tickFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;
markerIndices = unique(round(linspace(1, numel(testTime), 12)));
colorscheme = [
    0.00 0.45 0.74;
    0.00 0.60 0.20;
    0.85 0.10 0.10;
    0.75 0.00 0.75;
    0.00 1.00 1.00
];
set(groot, defaultTextInterpreter="latex");
set(groot, defaultLegendInterpreter="latex");
set(groot, defaultAxesTickLabelInterpreter="latex");

figure(Name="Feature-space Rollout Error"), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(liftedStateError(1, :, :)), ...
    LineWidth=2, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(gca,'YScale','log');
xlabel("Time [s]")
ylabel("$\left\|\hat{\bf{z}} - \bf{z}^\star\right\|$")
legend(predictorNames, Location="best")
if savePlots
    exportPlot(gcf(), 'data/feature_space_error.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="State-space Rollout Error"), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(scaledStateError(1, :, :)), ...
    LineWidth=2, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(gca,'YScale','log');
xlabel("Time [s]")
ylabel("$\left\|\hat{\bf{x}} - \bf{x}^\star\right\|$")
legend([memoryPredictorNames(:); predictorNames(:)], Location="best")
if savePlots
    exportPlot(gcf(), 'data/state_space_error.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="State Trajectories")
subplot(2, 1, 1), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(XPredicted(1, 1, :, :)), ...
    LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices ...
);
plot(testTime, squeeze(XTest{1}(1, :)), "k--", LineWidth=1.5)
ylabel("$x_1$ $[^\circ$C]")
legend([memoryPredictorNames(:); predictorNames(:); "True"], Location="best")

subplot(2, 1, 2), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(XPredicted(1, 2, :, :)), ...
    LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices ...
);
plot(testTime, squeeze(XTest{1}(2, :)), "k--", LineWidth=1.5)
xlabel("Time [s]")
ylabel("$x_2$ $[^\circ$C]")
legend([memoryPredictorNames(:); predictorNames(:); "True"], Location="best")
if savePlots
    exportPlot(gcf(), 'data/state_trajectories.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="Phase Portrait"), hold on, grid on, colororder(colorscheme)
for i = 1:numberOfPredictors
    plot(squeeze(XPredicted(1, 1, :, i)), ...
        squeeze(XPredicted(1, 2, :, i)), ...
        LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices);
end
plot(squeeze(XTest{1}(1, :)), squeeze(XTest{1}(2, :)), "k--", LineWidth=1.5)
plot(XTest{1}(1, 1), XTest{1}(2, 1), "ko", MarkerSize=10, MarkerFaceColor="y")
xlabel("$x_1$ $[^\circ$C]")
ylabel("$x_2$ $[^\circ$C]")
legend([memoryPredictorNames(:); predictorNames(:); "True"; "$x_0$"], Location="best")
if savePlots
    exportPlot(gcf(), 'data/phase_portrait.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

%% Save trained models
if saveModels
    fprintf("\n[6] Save trained models...\n");
    save(savePath, ...
        "tcLab", ...
        "stateScaling", "inputScaling", ...
        "stateDelayEmbedding", "inputDelayEmbedding", ...
        "stateLifting", "inputLifting", ...
        "predictors" ...
    );
    fprintf("\tSaved to %s\n", savePath);
end