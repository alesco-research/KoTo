%% Learn Koopman Operator for a TC-Lab
clc, clearvars, close all;
initWorkspace;
%#ok<*UNRCH>

%% Set parameters
rng(42); % for reproducibility
simulate = false;
savePlots = false;
saveModels = false;
plotTrainData = true;
plotTestData = true;
trainData = "TCLabTrain01.mat";
testData = "TCLabTest01.mat";

% Plotting parameters
margin = 0;
width = 15;
height = 10;
tickFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;
set(groot, defaultTextInterpreter="latex");
set(groot, defaultLegendInterpreter="latex");
set(groot, defaultAxesTickLabelInterpreter="latex");

%% Instantiate model, scaling, delay embedding and lifting
tcLab = TCLab(measurementNoiseCov=5E-2*[1, 0.5; 0.5, 1]);
[stateDim, inputDim] = deal(tcLab.stateDim, tcLab.inputDim);
stateScaling = Scale([23, 70; 23, 70]); % [x1] = °C, [x2] = °C
inputScaling = Scale([0, 100; 0, 100]); % [u1] = %, [u2] = %

[stateDelay, inputDelay] = deal(5, 5);
stateDelayEmbedding = DelayEmbedding(stateDim, stateDelay);
inputDelayEmbedding = DelayEmbedding(inputDim, inputDelay, 2);
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
inputLifting = PolynomialLift(inputDelayEmbedding.featureDim, 1);

switch "multi"
    case "one"
        getPairs = @(Z, U, V) getOneStepPairs(Z, U, V, maxNumberOfPairs=5000);
    case "multi"
        getPairs = @(Z, U, V) getMultiStepWindows(Z, U, V, horizon=100, maxNumberOfWindows=400);
end

% KCF coupling parameters
kcfCouplingRatio = 2;   % > 1
kcfCouplingDegree = 1;  % <= polynomial lift degree

dataDir = fullfile(fileparts(which("learnTCLab")), "data");
trainPath = fullfile(dataDir, trainData);
testPath = fullfile(dataDir, testData);
savePath = fullfile(dataDir, "TCLab" + upper(kernelType) + ".mat");

%% Load/generate training data
if ~simulate && isfile(trainPath)
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
    trainTime = (0:trainHorizon - 1)' * TsTrain; % time vector
    trainPeriods = 10:15:150; % driving input frequencies

    XTrain = cell(numberOfTrajectorySamples, 1);
    UTrain = cell(numberOfTrajectorySamples, 1);
    for l = 1:numberOfTrajectorySamples
        x0 = xRange(:,1) + diff(xRange, 1, 2) .* rand(2,1);
        uTrain = [
            1 + sum(sin(2 * pi * (trainTime ./ trainPeriods + rand(1,10))), 2)';
            1 + sum(sin(2 * pi * (trainTime ./ trainPeriods + rand(1,10))), 2)'
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
        XTrain, trajectoryOptions(labels={
            ["Temp. Heater 1", "$[^\circ$C]"];
            ["Temp. Heater 2", "$[^\circ$C]"]
        }), ...
        UTrain, trajectoryOptions(labels={
            ["Input Heater 1",  "[\%]"];
            ["Input Heater 2",  "[\%]"]
        }), ...
        dt = TsTrain ...
    );
    if savePlots
        exportPlot(gcf(), "data/train_data.png", ...
            width=width, ...
            height=height, ...
            margin=margin, ...
            tickFontSize=tickFontSize, ...
            labelFontSize=labelFontSize ...
        );
    end
end

%% Scaling
XTrainScaled = cellfun(@stateScaling.scale, XTrain, UniformOutput=0);
UTrainScaled = cellfun(@inputScaling.scale, UTrain, UniformOutput=0);

% Stack trajectories
X = cell2mat(cellfun(@(x) x(:, 1:end-1), XTrainScaled, UniformOutput=0)');
U = cell2mat(UTrainScaled');

%% Delay embedding
QTrain = cellfun(@stateDelayEmbedding.embed, XTrainScaled, UniformOutput=0);
WTrain = cellfun(@inputDelayEmbedding.embed, UTrainScaled, UniformOutput=0);

% Align embeddings
QTrain = cellfun(@(e) e(:, 1 + dqm:end), QTrain, UniformOutput=0);
WTrain = cellfun(@(e) e(:, 1 + dwm:end), WTrain, UniformOutput=0);

% Stack trajectories
Q = cell2mat(cellfun(@(q) q(:, 1:end-1), QTrain, UniformOutput=0)');
W = cell2mat(WTrain');

%% Lifting
fprintf("\n[2] Lift training data...\n");
if ~isa(stateLifting, "LiftBase")
    centers = [];
    if any(kernelType == ["rbf", "rq", "matern"])
        [~, centers] = kmeans(Q', numberOfCenters, MaxIter=300, Replicates=3);
    end
    stateLifting = stateLifting(centers');
end

% Lift trajectories
ZTrain = cellfun(@stateLifting.lift, QTrain, UniformOutput=0);
VTrain = cellfun(@inputLifting.lift, WTrain, UniformOutput=0);

% Stack trajectories
Z = cell2mat(cellfun(@(z) z(:, 1:end-1), ZTrain, UniformOutput=0)');
V = cell2mat(VTrain');

reprojectionError = stateLifting.fitProjection(Q, gamma=1e-4);
fprintf("\tReprojection Error: %.4f\n", reprojectionError);

%% Fit Koopman controllers
fprintf("\n[3] Fit Koopman controllers...\n");
[YPairs, ZPairs, WPairs, VPairs] = getPairs(ZTrain, WTrain, VTrain);
pairs = [
    cellfun(@(x) x(:,1:end-1), XTrainScaled(:)', UniformOutput=0);
    UTrainScaled(:)'
];
parameterBounds = struct( ...
    U=[1, 15], Us=[1, 50], m=[1E-3, 1E-2], Q1max=[.5, 1.5], Q2max=[.5, 1.5] ...
);

memoryPredictors = {};
memoryPredictors{end+1} = FundamentalLemma(pairs{:}, memory=10, horizon=40);
% memoryPredictors{end+1} = MOESP(pairs{:}, stateDim=4, L=50);
memoryPredictors{end+1} = NLSF( ...
     TCLab(Ts=TsTrain), parameterBounds, pairs{:}, ...
     outputScaling=stateScaling, ...
     inputScaling=inputScaling, ...
     horizon = inf ...
);
memoryPredictorNames = cellfun(@(c) string(class(c)), memoryPredictors);

predictors = {};
predictors{end+1} = LinearKoopman(YPairs, ZPairs, WPairs, gamma=5E-3);
predictors{end+1} = BilinearKoopman(YPairs, ZPairs, WPairs, gamma=5E-3);
predictors{end+1} = GeKoKoopman(YPairs, ZPairs, VPairs, gamma=5E-3);

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

predictors{end+1} = KCFKoopman(YPairs, ZPairs, VPairs, couplingIdxs=kcfCouplingIdxs, gamma=5E-3);
predictorNames = cellfun(@(c) erase(string(class(c)), "Koopman"), predictors);

% Evaluate operator norms
for i = 1:numel(predictors)
    fprintf("\t%s", evalc("predictors{" + i + "}.getOperatorNorm();"))
end

%% Load/generate test data
if ~simulate && isfile(testPath)
    fprintf("\n[4] Load test data...\n");
    load(testPath, "XTest", "UTest", "Ts")
    numberOfTestRuns = numel(XTest);
    testHorizons = cellfun(@(x) size(x, 2), XTest);
    testTime = arrayfun(@(len) (0:len - 1) * Ts, testHorizons, UniformOutput=0);

    if TsTrain ~= Ts
        error("Train and test sampling time mismatch.")
    end
else
    fprintf("\n[4] Generate testing trajectories...\n");
    testHorizons = [400; 400; 400];
    numberOfTestRuns = numel(testHorizons);

    testTime = arrayfun(@(len) (0:len - 1) * tcLab.Ts, testHorizons, UniformOutput=0);
    testPeriods = 10:15:150;

    XTest = cell(numberOfTestRuns, 1);
    UTest = cell(numberOfTestRuns, 1);
    for i = 1:numberOfTestRuns
        x0 = xRange(:,1) + diff(xRange, 1, 2) .* rand(2,1);
        uTest = [
            1 + sum(sin(2 * pi * (testTime{i}(1:end - 1)' ./ testPeriods + rand(1,10))), 2)';
            1 + sum(sin(2 * pi * (testTime{i}(1:end - 1)' ./ testPeriods + rand(1,10))), 2)'
        ] * 25;
        UTest{i} = clip(uTest, 0, 100);
        XTest{i} = tcLab.simulate(x0, UTest{i});
    end
end

% Scale trajectories
XTestScaled = cellfun(@stateScaling.scale, XTest, UniformOutput=0)';
UTestScaled = cellfun(@inputScaling.scale, UTest, UniformOutput=0)';

% Delay embedding
QTest = cellfun(@stateDelayEmbedding.embed, XTestScaled, UniformOutput=0);
WTest = cellfun(@inputDelayEmbedding.embed, UTestScaled, UniformOutput=0);

% Align embeddings
QTest = cellfun(@(e) e(:, 1 + dqm:end), QTest, UniformOutput=0);
WTest = cellfun(@(e) e(:, 1 + dwm:end), WTest, UniformOutput=0);

% Lift trajectories
ZTest = cellfun(@stateLifting.lift, QTest, UniformOutput=0);
VTest = cellfun(@inputLifting.lift, WTest, UniformOutput=0);

if plotTestData
    plotTrajectories( ...
        XTest, trajectoryOptions(labels={
            ["Temp. Heater 1", "$[^\circ$C]"];
            ["Temp. Heater 2", "$[^\circ$C]"]
        }), ...
        UTest, trajectoryOptions(labels={
            ["Input Heater 1",  "[\%]"];
            ["Input Heater 2",  "[\%]"]
        }) ...
    );
    if savePlots
        exportPlot(gcf(), "data/test_data.png", ...
            width=width, ...
            height=height, ...
            margin=margin, ...
            tickFontSize=tickFontSize, ...
            labelFontSize=labelFontSize ...
        );
    end
end

%% Evaluate prediction error
fprintf("\n[5] Evaluate prediction errors...\n");
numberOfPredictors = numel(memoryPredictors) + numel(predictors);

testInputs = {WTest, WTest, VTest, VTest};
XPredicted = arrayfun(@(len) nan(stateDim, len, numberOfPredictors), testHorizons, UniformOutput=0);
liftedStateError = arrayfun(@(len) nan(1, len, numel(predictors)), testHorizons, UniformOutput=0);
scaledStateError = arrayfun(@(len) nan(1, len, numberOfPredictors), testHorizons, UniformOutput=0);
for i = 1:numberOfTestRuns
    % State-full predictors
    for j = 1:numel(memoryPredictors)
        m = memoryPredictors{j}.memory;
        uWarm = UTestScaled{i}(:, 1:m);
        yWarm = XTestScaled{i}(:, 1:m);
        uFuture = UTestScaled{i}(:, m+1:end);
        
        XPredictedScaled = memoryPredictors{j}.forecast(uWarm, yWarm, uFuture);

        XPredicted{i}(:, m+1:end-1, j) = stateScaling.unscale(XPredictedScaled);
        scaledStateError{i}(1, m+1:end-1, j) = sqrt( ...
            sum((XPredictedScaled - XTestScaled{i}(:, m+1:end-1)).^2, 1, "omitnan") ...
        );
    end
    % Regular predictors
    for k = 1:numel(predictors)
        ZPredicted = predictors{k}.rollout(ZTest{i}(:, 1), testInputs{k}{i});
        
        QPredicted = stateLifting.project(ZPredicted);
        XPredictedScaled = stateDelayEmbedding.unembed(QPredicted);
        XPredicted{i}(:, memory+1:end, k+j) = stateScaling.unscale(XPredictedScaled);

        liftedStateError{i}(1, memory+1:end, k) = sqrt(sum((ZPredicted - ZTest{i}).^2, 1));
        scaledStateError{i}(1, memory+1:end, k+j) = sqrt( ...
            sum((XPredictedScaled - XTestScaled{i}(:, memory+1:end)).^2, 1) ...
        );
    end
end

meanLiftedStateErrors = cellfun(@(err) mean(err, 2, "omitnan"), liftedStateError, UniformOutput=0);
meanScaledStateErrors = cellfun(@(err) mean(err, 2, "omitnan"), scaledStateError, UniformOutput=0);

% Print MSE
fprintf("\tPer-trajectory mean error (scaled state)\n\tNr.");
fprintf("  %16s", [memoryPredictorNames(:); predictorNames(:)]);
fprintf("\n");
for i = 1:numberOfTestRuns
    fprintf("\t%-3d", i);
    fprintf("  %16.4e", meanScaledStateErrors{i});
    fprintf("\n");
end

%% Plot
fprintf("\n[6] Plotting...\n");
[~, axs] = plotTrajectories( ...
    liftedStateError, trajectoryOptions(labels={["$\left\|\hat{\bf{z}} - \bf{z}^\star\right\|$"]}), ...
    name="Feature-space Rollout Error", ...
    dt = TsTrain ...
);
grid(axs, "on")
set(axs,'YScale','log');
legend(axs, predictorNames, Location="best")
if savePlots
    exportPlot(gcf(), "data/feature_space_error.png", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

[~, axs] = plotTrajectories( ...
    scaledStateError, trajectoryOptions(labels={["$\left\|\hat{\bf{x}} - \bf{x}^\star\right\|$"]}), ...
    name="State-space Rollout Error", ...
    dt = TsTrain ...
);
grid(axs, "on")
set(axs,'YScale','log');
legend(axs, [memoryPredictorNames(:); predictorNames(:)], Location="best")
if savePlots
    exportPlot(gcf(), "data/state_space_error.png", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

[~, axs] = plotTrajectories( ...
    XPredicted, trajectoryOptions(labels={
            ["Temp. Heater 1", "$[^\circ$C]"];
            ["Temp. Heater 2", "$[^\circ$C]"]
    }), ...
    name="State Trajectories", ...
    dt = TsTrain ...
);
plotTrajectories(XTest, trajectoryOptions(lineStyle="--", color="k"), axs=axs, dt=TsTrain);
grid(axs, "on")
legend(axs(1), [memoryPredictorNames(:); predictorNames(:); "True"], Location="eastoutside")
if savePlots
    exportPlot(gcf(), "data/state_trajectories.png", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="Phase Portrait"), hold on, grid on, colororder("gem12")
for i = 1:numberOfPredictors
    plot(squeeze(XPredicted{1}(1, :, i)), ...
        squeeze(XPredicted{1}(2, :, i)), ...
        LineWidth=1.5, MarkerSize=5);
end
plot(squeeze(XTest{1}(1, :)), squeeze(XTest{1}(2, :)), "k--", LineWidth=1.5)
plot(XTest{1}(1, 1), XTest{1}(2, 1), "ko", MarkerSize=10, MarkerFaceColor="y")
xlabel("$x_1$ $[^\circ$C]")
ylabel("$x_2$ $[^\circ$C]")
legend([memoryPredictorNames(:); predictorNames(:); "True"; "$x_0$"], Location="best")
if savePlots
    exportPlot(gcf(), "data/phase_portrait.png", ...
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