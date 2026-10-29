%% Learn Koopman Operator for a TC-Lab
clc, clearvars, close all;
initWorkspace;
%#ok<*UNRCH>

%% Set parameters
rng(42); % for reproducibility
simulate = false;
savePlots = true;
saveModels = false;
plotTrainData = true;
plotTestData = true;
trainData = "TCLabTrain01.mat";
testData = "TCLabTest01.mat";

% Plotting parameters
margin = 0;
width = 8.6; % [cm]
height = 5; % [cm]
tickFontSize = 8;
labelFontSize = 10;
legendFontSize = 6;
legendLineLength = 10;
set(groot, defaultLineLineWidth=1);
set(groot, defaultLineMarkerSize=2);
set(groot, defaultTextInterpreter="latex");
set(groot, defaultLegendInterpreter="latex");
set(groot, defaultAxesTickLabelInterpreter="latex");

%% Instantiate model, scaling, delay embedding and lifting
tcLab = TCLab(measurementNoiseCov=5E-2*[1, 0.5; 0.5, 1]);
[stateDim, inputDim] = deal(tcLab.stateDim, tcLab.inputDim);
stateScaling = Scale([23, 70; 23, 70]); % [x1] = °C, [x2] = °C
inputScaling = Scale([0, 100; 0, 100]); % [u1] = %, [u2] = %

[stateDelay, inputDelay] = deal(0, 12);
stateDelayEmbedding = DelayEmbedding(stateDim, stateDelay, 2);
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
        stateLifting = @(centers) Matern52Lift(stateDelayEmbedding.featureDim, 0.3, centers);
        numberOfCenters = 300;
    case "poly"
        stateLifting = @(~) PolynomialLift(stateDelayEmbedding.featureDim, 3);
end
inputLifting = PolynomialLift(inputDelayEmbedding.featureDim, 1);

switch "one"
    case "one"
        getPairs = @(Z, U, V) getOneStepPairs(Z, U, V, maxNumberOfPairs=40000);
    case "multi"
        getPairs = @(Z, U, V) getMultiStepWindows(Z, U, V, horizon=70, maxNumberOfWindows=800);
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
    fig = plotTrajectories( ...
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
        exportPlot(fig, "data/train_data.png", ...
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

memoryPredictors = {};
memoryPredictors{end+1} = FundamentalLemma(pairs{:}, memory=10, horizon=40);
memoryPredictorNames = {"Willem's"};

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
    fig = plotTrajectories( ...
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
        exportPlot(fig, "data/test_data.png", ...
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
liftedStateError = arrayfun(@(len) nan(2, len, numberOfPredictors), testHorizons, UniformOutput=0);
scaledStateError = arrayfun(@(len) nan(2, len, numberOfPredictors), testHorizons, UniformOutput=0);
for i = 1:numberOfTestRuns
    % State-full predictors
    for j = 1:numel(memoryPredictors)
        m = memoryPredictors{j}.memory;

        % Multi-step predictions
        [uWarm, uFuture] = deal(UTestScaled{i}(:, 1:m), UTestScaled{i}(:, m+1:end));
        yWarm = XTestScaled{i}(:, 1:m);
        
        XPredictedScaled = memoryPredictors{j}.forecast(uWarm, yWarm, uFuture);

        XPredicted{i}(:, m+1:end-1, j) = stateScaling.unscale(XPredictedScaled);
        scaledStateError{i}(2, m+1:end-1, j) = sqrt( ...
            sum((XPredictedScaled - XTestScaled{i}(:, m+1:end-1)).^2, 1, "omitnan") ...
        );

        % Lift predicted states (warm-start window provides the delays)
        QPredicted = stateDelayEmbedding.embed([yWarm, XPredictedScaled]);
        ZPredicted = stateLifting.lift(QPredicted(:, 1 + dqm:end));
        liftedStateError{i}(2, memory+1:end-1, j) = sqrt( ...
            sum((ZPredicted - ZTest{i}(:, 1:end-1)).^2, 1, "omitnan") ...
        );
        liftedStateError{i}(2, 1:m, j) = nan;

        % One-step prediction errors
        for t = m+1:size(UTestScaled{i}, 2)
            xNextScaled = memoryPredictors{j}.forecast( ...
                UTestScaled{i}(:, t-m:t-1), XTestScaled{i}(:, t-m:t-1), UTestScaled{i}(:, t) ...
            );
            scaledStateError{i}(1, t, j) = sqrt( ...
                sum((xNextScaled - XTestScaled{i}(:, t)).^2, 1, "omitnan") ...
            );
        end
    end
    % Koopman predictors
    for k = 1:numel(predictors)
        % Multi-step predictions
        ZPredicted = predictors{k}.rollout(ZTest{i}(:, 1), testInputs{k}{i});
        
        QPredicted = stateLifting.project(ZPredicted);
        XPredictedScaled = stateDelayEmbedding.unembed(QPredicted);
        XPredicted{i}(:, memory+1:end, k+j) = stateScaling.unscale(XPredictedScaled);

        liftedStateError{i}(2, memory+1:end, k+j) =sqrt(sum((ZPredicted - ZTest{i}).^2, 1));
        scaledStateError{i}(2, memory+1:end, k+j) = sqrt( ...
            sum((XPredictedScaled - XTestScaled{i}(:, memory+1:end)).^2, 1) ...
        );

        % One-step prediction error
        ZOneStep = predictors{k}.predict(ZTest{i}(:, 1:end-1), testInputs{k}{i});
        XOneStepScaled = stateDelayEmbedding.unembed(stateLifting.project(ZOneStep));

        liftedStateError{i}(1, memory+2:end, k+j) =sqrt(sum((ZOneStep - ZTest{i}(:, 2:end)).^2, 1));
        scaledStateError{i}(1, memory+2:end, k+j) = sqrt( ...
            sum((XOneStepScaled - XTestScaled{i}(:, memory+2:end)).^2, 1) ...
        );
    end
end

meanLiftedStateErrors = cellfun(@(err) mean(err, 2, "omitnan"), liftedStateError, UniformOutput=0);
meanScaledStateErrors = cellfun(@(err) mean(err, 2, "omitnan"), scaledStateError, UniformOutput=0);

% Print mean error norms
fprintf("\tPer-trajectory single-step mean error norm\n\tNr.");
fprintf("  %16s", [memoryPredictorNames(:); predictorNames(:)]);
fprintf("\n");
for i = 1:numberOfTestRuns
    fprintf("\t%-3d", i);
    fprintf("  %16.4e", meanScaledStateErrors{i}(1,:));
    fprintf("\n");
end

fprintf("\n\tPer-trajectory multi-step mean error norm\n\tNr.");
fprintf("  %16s", [memoryPredictorNames(:); predictorNames(:)]);
fprintf("\n");
for i = 1:numberOfTestRuns
    fprintf("\t%-3d", i);
    fprintf("  %16.4e", meanScaledStateErrors{i}(2,:));
    fprintf("\n");
end

%% Plot
fprintf("\n[6] Plotting...\n");
[fig, axs] = plotTrajectories( ...
    liftedStateError, trajectoryOptions(labels={
        ["One-Step Error", "$\left\|\hat{\bf{z}} - \bf{z}^\star\right\|$"];
        ["Multi-Step Error", "$\left\|\hat{\bf{z}} - \bf{z}^\star\right\|$"]
    }), ...
    name="Feature-space Rollout Error", ...
    dt = TsTrain ...
);
grid(axs, "on")
set(axs,'YScale','log');
lgd = legend(axs(1), [memoryPredictorNames(:); predictorNames(:)], Orientation="horizontal");
headerLegend(lgd, ...
    width=width, ...
    height=height, ...
    fontSize=legendFontSize, ...
    lineLength=legendLineLength ...
);
if savePlots
    exportPlot(fig, "data/feature_space_error.pdf", ...
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
    cellfun(@(err) err(2,:,:), scaledStateError, UniformOutput=0), ...
    trajectoryOptions(labels={
        ["$\left\|\hat{\bf{x}} - \bf{x}^\star\right\|$"]
        }), ...
    name="State-space Rollout Error", ...
    dt = TsTrain ...
);
grid(axs, "on")
set(axs,'YScale','log');
legend(axs(1), [memoryPredictorNames(:); predictorNames(:)], Location="southeast");
if savePlots
    exportPlot(fig, "data/state_space_error.pdf", ...
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
lgd = legend(axs(1), [memoryPredictorNames(:); predictorNames(:); "True"], Orientation="horizontal");
headerLegend(lgd, ...
    width=width, ...
    height=height, ...
    fontSize=legendFontSize, ...
    lineLength=legendLineLength ...
);
if savePlots
    exportPlot(fig, "data/state_trajectories.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, ...
        legendLineLength=legendLineLength ...
    );
end

figure(Name="Phase Portrait"), hold on, grid on, colororder("gem12")
for i = 1:numberOfPredictors
    plot(squeeze(XPredicted{1}(1, :, i)), squeeze(XPredicted{1}(2, :, i)));
end
plot(squeeze(XTest{1}(1, :)), squeeze(XTest{1}(2, :)), "k--", LineWidth=1)
plot(XTest{1}(1, 1), XTest{1}(2, 1), "ko", MarkerSize=10, MarkerFaceColor="y")
xlabel("$x_1$ $[^\circ$C]")
ylabel("$x_2$ $[^\circ$C]")
lgd = legend([memoryPredictorNames(:); predictorNames(:); "True"; "$x_0$"], Location="southeast");
if savePlots
    exportPlot(gcf(), "data/phase_portrait.pdf", ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize, ...
        legendFontSize=legendFontSize, ...
        legendLineLength=legendLineLength ...
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