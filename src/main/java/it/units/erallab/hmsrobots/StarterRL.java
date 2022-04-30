package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredControlFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredObservationFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.RLController;
import it.units.erallab.hmsrobots.core.controllers.rl.RewardFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.TabularSARSALambda;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryInputConverter;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryOutputConverter;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.rllocomotion.RLEnsembleOutcome;
import it.units.erallab.hmsrobots.tasks.rllocomotion.RLLocomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.ToDoubleFunction;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;

public class StarterRL {

    public static double[][] getTerrain() {
        double TERRAIN_BORDER_WIDTH = 10d;
        double TERRAIN_BORDER_HEIGHT = 100d;
        int TERRAIN_LENGTH = 1000000;
        return new double[][]{new double[]{0, TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH - TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH}, new double[]{TERRAIN_BORDER_HEIGHT, 5, 5, TERRAIN_BORDER_HEIGHT}};
    }

    public static void main(String[] args) {

        int nThreads = Integer.parseInt(args[0]);
        String shape = args[1];
        double controllerStep = Double.parseDouble(args[2]);

        ExecutorService executor = Executors.newFixedThreadPool(nThreads);
        List<Callable<Integer>> callables = new ArrayList<>();

        for (int i = 0; i < nThreads; i++) {
            int finalI = i;
            callables.add(() -> {
                runTabularSARSALambda(finalI, shape, false, true, controllerStep);
                return 0;
            });
        }

        try {
            for (Future<Integer> future : executor.invokeAll(callables)) {
                future.get();
            }
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }

        executor.shutdown();
    }

    private static void runTabularSARSALambda(
            int seed, String shape, boolean touch, boolean area, double controllerStep
    ) {

        // Configs
        String sensorConfig = "uniform-a+t+r+vxy-0";
        int nClusters = 4;

        // Create the body and the clusters
        Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(RobotUtils.buildShape(shape));
        Set<Set<Grid.Key>> clustersSet = computeCardinalPoses(Grid.create(body, Objects::nonNull));
        List<List<Grid.Key>> clusters = clustersSet.stream().map(s -> s.stream().toList()).toList();

        // Create observation function
        ClusteredObservationFunction observationFunction = new ClusteredObservationFunction(clusters, area, touch, true);

        // Compute dimensions
        int sensorReadingsDimension = observationFunction.getOutputDimension();
        int actionSpaceDimension = (int) Math.pow(2, nClusters);
        int stateSpaceDimension = (int) Math.pow(2, sensorReadingsDimension);

        // Create the reward function
        ToDoubleFunction<Grid<Voxel>> rewardFunction = new RewardFunction();

        // Create binary input converter
        DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(sensorReadingsDimension);

        // Create binary output converter
        DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(nClusters, 0.5);

        // Create Tabular Q-Learning agent
        TabularSARSALambda rlAgentDiscrete = new TabularSARSALambda(
                0.95,
                0.8,
                stateSpaceDimension,
                actionSpaceDimension,
                0d,
                0.05,
                seed
        );

        // Create continuous agent from discrete one
        ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, binaryOutputConverter);

        // Create control function
        ClusteredControlFunction controlFunction = new ClusteredControlFunction(clusters);

        // Create the RL controller and apply it to the body
        RLController rlController = new RLController(observationFunction, rewardFunction, rlAgent, controlFunction);
        StepController stepController = new StepController(rlController, controllerStep);
        Robot robot = new Robot(stepController, SerializationUtils.clone(body));


        // Create the environment
        //for (int i = 0; i < 100; i++) {
        //    Locomotion locomotion = new Locomotion(50, getTerrain(), 10000, new Settings());
        //    Outcome outcome = locomotion.apply(robot);
        //    System.out.println(outcome.getDistance() / outcome.getTime());
        //}

        //File file = new File("results.csv");
        //listener.toFile(file);
        //System.out.println("Results saved to " + file.getAbsolutePath());
        //Locomotion locomotion = new Locomotion(60, getTerrain(), 10000, new Settings());
        //GridOnlineViewer.run(locomotion, Grid.create(1, 1, new NamedValue<>("SARSA", robot)), Drawers::basicWithMiniWorldAndRL);

        RLLocomotion rlLocomotion = new RLLocomotion(1500d, 50d, 1, robot);
        RLEnsembleOutcome outcome = rlLocomotion.apply(rewardFunction);
        System.out.println(Arrays.toString(outcome.results().stream().map(RLEnsembleOutcome.RLOutcome::validationVelocity).toArray()));
    }

}
