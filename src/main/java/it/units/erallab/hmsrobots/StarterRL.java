package it.units.erallab.hmsrobots;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.*;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.TabularQLearning;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.TabularSARSALambda;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryInputConverter;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryOutputConverter;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.TabularQLambda;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.*;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.tasks.rllocomotion.RLEnsembleOutcome;
import it.units.erallab.hmsrobots.tasks.rllocomotion.RLLocomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
import it.units.erallab.hmsrobots.viewers.NamedValue;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.erallab.hmsrobots.viewers.drawers.Drawers;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;
import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeClusteredByPositionPoses;

public class StarterRL {

  public static double[][] getTerrain() {
    double TERRAIN_BORDER_WIDTH = 10d;
    double TERRAIN_BORDER_HEIGHT = 100d;
    int TERRAIN_LENGTH = 1000000;
    return new double[][]{new double[]{0, TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH - TERRAIN_BORDER_WIDTH, TERRAIN_LENGTH}, new double[]{TERRAIN_BORDER_HEIGHT, 5, 5, TERRAIN_BORDER_HEIGHT}};
  }

  public static void main(String[] args) {

    int nThreads = 1;
    int seed = Integer.parseInt(args[0]);
    String shape = args[1];
    double actuatorsForce = Double.parseDouble(args[2]);
    double controllerFreq = Double.parseDouble(args[3]);

    ExecutorService executor = Executors.newFixedThreadPool(nThreads);
    List<Callable<Integer>> callables = new ArrayList<>();

    for (int i = 0; i < nThreads; i++) {
      int finalI = i;
      callables.add(() -> {
        System.out.println("Thread " + finalI + " started");
        runRL(seed, shape, actuatorsForce, controllerFreq);
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
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(Grid.create(5,
        4,
        (x, y) -> (y == 0 && (x == 0 || x == 3)) || ((y == 1 || y == 2) && x < 4) || (y == 3 && (x == 3 || x == 4))
    ));
    Set<Set<Grid.Key>> clustersSet = computeClusteredByPositionPoses(Grid.create(body, Objects::nonNull), 4, 99);
    List<List<Grid.Key>> clusters = clustersSet.stream().map(s -> s.stream().toList()).toList();

    // Create observation function
    ClusteredObservationFunction.Config cfg = new ClusteredObservationFunction.Config(false,
        false,
        true,
        true,
        true,
        false,
        false
    );
    ClusteredObservationFunction observationFunction = new ClusteredObservationFunction(clusters, cfg);

    // Compute dimensions
    int sensorReadingsDimension = observationFunction.getOutputDimension();
    int actionSpaceDimension = (int) Math.pow(2, nClusters);
    int stateSpaceDimension = (int) Math.pow(2, sensorReadingsDimension);

    // Create the reward function
    RewardFunction rewardFunction = new RewardFunction() {

      @JsonProperty
      private double previousPosition = Double.NEGATIVE_INFINITY;

      @Override
      public Double apply(Grid<Voxel> voxels) {

        if (previousPosition == Double.NEGATIVE_INFINITY) {
          previousPosition = voxels.get(0, 0).center().x();
        }

        double rotation = voxels.get(0, 0).getAngle();
        double currentPosition = voxels.get(0, 0).center().x();
        double deltaPosition = currentPosition - previousPosition;

        double reward = rotation < -Math.PI / 2 || rotation > Math.PI / 2 ? -50d : (deltaPosition <= 0d ? -25d : 10 * deltaPosition);

        previousPosition = currentPosition;

        return reward;
      }

      @Override
      public <V> Function<V, Double> compose(Function<? super V, ? extends Grid<Voxel>> before) {
        return RewardFunction.super.compose(before);
      }

      @Override
      public <V> Function<Grid<Voxel>, V> andThen(Function<? super Double, ? extends V> after) {
        return RewardFunction.super.andThen(after);
      }
    };

    // Create binary input converter
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(sensorReadingsDimension);

    // Create binary output converter
    DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(nClusters, 0.5);

    // Create Tabular Q-Learning agent
    TabularQLambda rlAgentDiscrete = new TabularQLambda(0.95,
        0.5,
        stateSpaceDimension,
        actionSpaceDimension,
        0d,
        0.05d,
        seed
    );

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, binaryOutputConverter);

    // Create control function
    ClusteredControlFunction controlFunction = new ClusteredControlFunction(clusters);

    // Create the RL controller and apply it to the body
    RLController rlController = new RLController(observationFunction, rewardFunction, rlAgent, controlFunction);
    StepController stepController = new StepController(rlController, controllerStep);
    //SmoothedController smoothedController = new SmoothedController(stepController, 3.75);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    Locomotion locomotion = new Locomotion(60, getTerrain(), 10000, new Settings());
    GridOnlineViewer.run(locomotion, Grid.create(1, 1, new NamedValue<>("SARSA", robot)));

    // Create the environment
    //RLLocomotion locomotion = new RLLocomotion(10000, 100, 1, robot);
    //RLEnsembleOutcome outcome = locomotion.apply(rewardFunction);
    //System.out.println(outcome.results().stream().map(RLEnsembleOutcome.RLOutcome::validationVelocity).collect(Collectors.toList()));

    // Create the environment
    //Locomotion locomotion = new Locomotion(50, getTerrain(), 10000, new Settings());
    //Grid

    //File file = new File("results.csv");
    //listener.toFile(file);
    //System.out.println("Results saved to " + file.getAbsolutePath());
    Locomotion locomotionTest = new Locomotion(45, getTerrain(), 10000, new Settings());
    GridOnlineViewer.run(locomotionTest, Grid.create(1, 1, new NamedValue<>("SARSA", robot)));

    //RLLocomotion rlLocomotion = new RLLocomotion(1500d, 50d, 1, robot);
    //RLEnsembleOutcome outcome = rlLocomotion.apply(rewardFunction);
    //System.out.println(Arrays.toString(outcome.results().stream().map(RLEnsembleOutcome.RLOutcome::validationVelocity).toArray()));
  }


  private static void distributedRL(int seed, String shape, boolean touch, boolean area, double controllerStep) {

    // Configs
    String sensorConfig = "uniform-a+Ax+Ay+Vx+Vy+t+r+vxy-0";
    int nClusters = 4;

    // Create the body and the clusters
    //Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(Grid.create(5, 4,
    //        (x, y) -> (y == 0 && (x == 0 || x == 3)) || ((y == 1 || y == 2) && x < 4 ) || (y == 3 && (x == 3 || x == 4))));
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(RobotUtils.buildShape("biped-4x3"));
    Set<Set<Grid.Key>> clustersSet = computeClusteredByPositionPoses(Grid.create(body, Objects::nonNull), 10, 99);
    List<List<Grid.Key>> clusters = clustersSet.stream().map(s -> s.stream().toList()).toList();

    System.out.println(clusters);

    // Create the reward function
    RewardFunction rewardFunction = new RewardFunction() {

      @JsonProperty
      private double previousPosition = Double.NEGATIVE_INFINITY;

      @Override
      public Double apply(Grid<Voxel> voxels) {

        if (previousPosition == Double.NEGATIVE_INFINITY) {
          previousPosition = voxels.get(0, 0).center().x();
        }

        double rotation = voxels.get(0, 0).getAngle();
        double currentPosition = voxels.get(0, 0).center().x();
        double deltaPosition = currentPosition - previousPosition;

        double reward = rotation < -Math.PI / 2 || rotation > Math.PI / 2 ? -50d : (deltaPosition <= 0.1 ? -25d : 10 * deltaPosition);

        previousPosition = currentPosition;

        return reward;
      }

      @Override
      public <V> Function<V, Double> compose(Function<? super V, ? extends Grid<Voxel>> before) {
        return RewardFunction.super.compose(before);
      }

      @Override
      public <V> Function<Grid<Voxel>, V> andThen(Function<? super Double, ? extends V> after) {
        return RewardFunction.super.andThen(after);
      }
    };

    Random random = new Random(100);
    List<ClusteredObservationFunction> obs = new ArrayList<>();
    List<ContinuousRL> rls = new ArrayList<>();

    for (int i = 0; i < clusters.size(); i++) {

      List<List<Grid.Key>> subClusters = new ArrayList<>();
      subClusters.add(clusters.get(i));
      Grid.Key key = clusters.get(i).get(0);
      if (body.get(key.x() + 1, key.y()) != null) {
        List<Grid.Key> tmp = new ArrayList<>();
        tmp.add(new Grid.Key(key.x() + 1, key.y()));
        subClusters.add(tmp);
      }
      if (body.get(key.x() - 1, key.y()) != null) {
        List<Grid.Key> tmp = new ArrayList<>();
        tmp.add(new Grid.Key(key.x() - 1, key.y()));
        subClusters.add(tmp);
      }
      if (body.get(key.x(), key.y() + 1) != null) {
        List<Grid.Key> tmp = new ArrayList<>();
        tmp.add(new Grid.Key(key.x(), key.y() + 1));
        subClusters.add(tmp);
      }
      if (body.get(key.x(), key.y() - 1) != null) {
        List<Grid.Key> tmp = new ArrayList<>();
        tmp.add(new Grid.Key(key.x(), key.y() - 1));
        subClusters.add(tmp);
      }

      // Create observation function
      ClusteredObservationFunction.Config cfg = new ClusteredObservationFunction.Config(false,
          false,
          false,
          false,
          false,
          true,
          true
      );
      ClusteredObservationFunction observationFunction = new ClusteredObservationFunction(subClusters, cfg);
      obs.add(observationFunction);

      System.out.println("output dim:" + observationFunction.getOutputDimension());

      // Compute dimensions
      int sensorReadingsDimension = observationFunction.getOutputDimension();
      int actionSpaceDimension = (int) Math.pow(2, 1);
      int stateSpaceDimension = (int) Math.pow(2, sensorReadingsDimension);

      // Create binary input converter
      DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(sensorReadingsDimension);

      // Create binary output converter
      DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(1, 0.5);

      // Create Tabular Q-Learning agent
      TabularQLearning rlAgentDiscrete = new TabularQLearning(0.95,
          stateSpaceDimension,
          actionSpaceDimension,
          0d,
          0.05d,
          seed
      );

      // Create continuous agent from discrete one
      ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, binaryOutputConverter);
      rls.add(rlAgent);
    }

    // Create control function
    ClusteredControlFunction controlFunction = new ClusteredControlFunction(clusters);

    // Create the RL controller and apply it to the body
    DistributedRLController rlController = new DistributedRLController(obs, rewardFunction, rls, controlFunction);
    StepController stepController = new StepController(rlController, 0.35);
    //SmoothedController smoothedController = new SmoothedController(stepController, 5d);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    //Locomotion locomotion2 = new Locomotion(60, getTerrain(), 10000, new Settings());
    //GridOnlineViewer.run(locomotion2, Grid.create(1, 1, new NamedValue<>("SARSA", robot)));
    //int a = System.in.read();
    //System.exit(0);

    // Create the environment
    for (int i = 0; i < 40; i++) {
      Locomotion locomotion = new Locomotion(75, getTerrain(), 10000, new Settings());
      Outcome outcome = locomotion.apply(robot);
    }
    // Create the environment
    //Locomotion locomotion = new Locomotion(50, getTerrain(), 10000, new Settings());
    //Grid

    //File file = new File("results.csv");
    //listener.toFile(file);
    //System.out.println("Results saved to " + file.getAbsolutePath());
    Locomotion locomotionTest = new Locomotion(60, getTerrain(), 10000, new Settings());
    GridFileWriter.save(locomotionTest,
        Grid.create(1, 1, new NamedValue<>("robot", robot)),
        600,
        600,
        5,
        20,
        VideoUtils.EncoderFacility.JCODEC,
        new File("012distributedvalidation.mp4"),
        Drawers::basicWithMiniWorld
    );
    //RLLocomotion rlLocomotion = new RLLocomotion(1500d, 50d, 1, robot);
    //RLEnsembleOutcome outcome = rlLocomotion.apply(rewardFunction);
    //System.out.println(Arrays.toString(outcome.results().stream().map(RLEnsembleOutcome.RLOutcome::validationVelocity).toArray()));
  }


  private static void runRL(int seed, String shape, double actuatorsForce, double controllerFreq) {

    // Configs
    int nClusters = 4;
    String sensorConfig = "uniform-a+t+r+vxy+Vx+Vy+Ax+Ay-0";

    // Create the body and the clusters
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(RobotUtils.buildShape(shape));
    Set<Set<Grid.Key>> clustersSet = computeCardinalPoses(Grid.create(body, Objects::nonNull));
    List<List<Grid.Key>> clusters = clustersSet.stream().map(s -> s.stream().toList()).toList();

    // Create observation function map
    LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map = new LinkedHashMap<>();

    // Touch sensors
    List<Grid.Key> foot1 = new ArrayList<>();
    foot1.add(new Grid.Key(0, 0));
    List<Grid.Key> foot2 = new ArrayList<>();
    foot2.add(new Grid.Key(3, 0));
    LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>> footMap = new LinkedHashMap<>();
    footMap.put(Touch.class, x -> Arrays.stream(x).max().orElse(0d));
    map.put(foot1, footMap);
    map.put(foot2, footMap);

    // Rotation and acceleration sensors
    List<Grid.Key> center = new ArrayList<>();
    center.add(new Grid.Key(1, 1));
    LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>> centerMap = new LinkedHashMap<>();
    centerMap.put(Angle.class, x -> x.length == 0 ? 0d : Arrays.stream(x).sum() / x.length);
    centerMap.put(AccelerationX.class, x -> x.length == 0 ? 0d : Arrays.stream(x).sum() / x.length);
    centerMap.put(AccelerationY.class, x -> x.length == 0 ? 0d : Arrays.stream(x).sum() / x.length);
    map.put(center, centerMap);

    // Create observation function
    ClusteredObservationFunction.Config cfg = new ClusteredObservationFunction.Config(false,
        false,
        false,
        false,
        false,
        false,
        false
    );
    ClusteredObservationFunction observationFunction = new ClusteredObservationFunction(clusters, cfg);
    observationFunction.setMap(map);

    // Compute dimensions
    int sensorReadingsDimension = observationFunction.getOutputDimension();
    int actionSpaceDimension = (int) Math.pow(2, nClusters);
    int stateSpaceDimension = (int) Math.pow(2, sensorReadingsDimension);
    System.out.println("State space dimension: " + stateSpaceDimension);
    System.out.println("Action space dimension: " + actionSpaceDimension);
    System.out.println("Sensor readings dimension: " + sensorReadingsDimension);

    // Create the reward function
    RewardFunction rewardFunction = new RewardFunction() {

      @JsonProperty
      private double previousPosition = Double.NEGATIVE_INFINITY;

      @Override
      public Double apply(Grid<Voxel> voxels) {

        if (previousPosition == Double.NEGATIVE_INFINITY) {
          previousPosition = voxels.get(1, 1).center().x();
        }

        double rotation = voxels.get(1, 1).getAngle();
        double currentPosition = voxels.get(1, 1).center().x();
        double deltaPosition = currentPosition - previousPosition;

        double reward = rotation < -Math.PI / 2 || rotation > Math.PI / 2 ? -50d : (deltaPosition <= 0d ? -25d : 10 * deltaPosition);

        previousPosition = currentPosition;

        return reward;
      }

      @Override
      public <V> Function<V, Double> compose(Function<? super V, ? extends Grid<Voxel>> before) {
        return RewardFunction.super.compose(before);
      }

      @Override
      public <V> Function<Grid<Voxel>, V> andThen(Function<? super Double, ? extends V> after) {
        return RewardFunction.super.andThen(after);
      }
    };

    // Create binary input converter
    DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(sensorReadingsDimension);

    // Create binary output converter
    DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(nClusters, actuatorsForce);

    // Create Tabular Q-Learning agent
    TabularSARSALambda rlAgentDiscrete = new TabularSARSALambda(0.95,
        0.75,
        stateSpaceDimension,
        actionSpaceDimension,
        0d,
        0.1d,
        seed
    );

    // Create continuous agent from discrete one
    ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, binaryOutputConverter);

    // Create control function
    ClusteredControlFunction controlFunction = new ClusteredControlFunction(clusters);

    // Create the RL controller and apply it to the body
    RLController rlController = new RLController(observationFunction, rewardFunction, rlAgent, controlFunction);
    StepController stepController = new StepController(rlController, controllerFreq);
    Robot robot = new Robot(stepController, SerializationUtils.clone(body));

    //Locomotion locomotion = new Locomotion(60, getTerrain(), 10000, new Settings());
    //GridOnlineViewer.run(locomotion, Grid.create(1, 1, new NamedValue<>("SARSA", robot)), Drawers::basicWithMiniWorldAndRL);

    // Create the environment
    RLLocomotion locomotion = new RLLocomotion(10000, 100, 1, robot);
    RLEnsembleOutcome outcome = locomotion.apply(rewardFunction);
    System.out.println(outcome.results()
        .stream()
        .map(RLEnsembleOutcome.RLOutcome::validationVelocity)
        .collect(Collectors.toList()));

    // Create the environment
    //Locomotion locomotion = new Locomotion(50, getTerrain(), 10000, new Settings());
    //Grid

    //File file = new File("results.csv");
    //listener.toFile(file);
    //System.out.println("Results saved to " + file.getAbsolutePath());
    //Locomotion locomotionTest = new Locomotion(45, getTerrain(), 10000, new Settings());
    //GridOnlineViewer.run(locomotionTest, Grid.create(1, 1, new NamedValue<>("SARSA", robot)));

    //RLLocomotion rlLocomotion = new RLLocomotion(1500d, 50d, 1, robot);
    //RLEnsembleOutcome outcome = rlLocomotion.apply(rewardFunction);
    //System.out.println(Arrays.toString(outcome.results().stream().map(RLEnsembleOutcome.RLOutcome::validationVelocity).toArray()));
  }


}
