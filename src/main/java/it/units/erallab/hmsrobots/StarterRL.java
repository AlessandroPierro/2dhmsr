package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.behavior.PoseUtils;
import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.*;
import it.units.erallab.hmsrobots.core.controllers.rl.tabular.TabularSARSALambda;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
import it.units.erallab.hmsrobots.viewers.NamedValue;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.erallab.hmsrobots.viewers.drawers.Drawers;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class StarterRL {

  private record Experiment(String name, String datetime, String shape) {}

  static final String RESULTS_PATH = "/home/alessandro/experiments/";

  public static double[][] createTerrain() {
    return new double[][]{new double[]{0, 10d, 5000d - 10d, 5000d}, new double[]{100d, 5, 5, 100d}};
  }


  static Predicate<Map<Double, Outcome.Observation>> makeStoppingCriterion(double remainingTime) {
    return map -> {
      if (map.isEmpty()) {
        return false;
      }
      double lastTime = map.keySet().stream().max(Double::compareTo).orElse(-1d);
      if (lastTime > remainingTime) {
        return true;
      }
      map = map.entrySet()
          .stream()
          .filter(e -> e.getKey() >= lastTime - 2)
          .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
      return map.values()
          .stream()
          .map(obs -> obs.voxelPolies()
              .stream()
              .filter(e -> e.key().y() == 0 && e.value() != null)
              .map(Grid.Entry::value)
              .findFirst()
              .get()
              .getAngle())
          .map(angle -> angle < -Math.PI / 2d || angle > Math.PI / 2d)
          .reduce(true, (a, b) -> a && b);
    };
  }


  static void runBipedTabularSARSA() throws IOException {

    int nInputPartitions = 2;
    int nOutputPartitions = 4;
    double maxAppliedForce = 0.75;

    double discountFactor = 0.95;
    double meanQ = 0d;
    double stdQ = 0.1;
    int seed = 42;
    double lambda = 0.75;

    // Create a biped
    Grid<Boolean> morphology = RobotUtils.buildShape("biped-4x3");
    Grid<Voxel> body = RobotUtils.buildSensorizingFunction("uniform-ax+t+r+vxy-0").apply(morphology);

    // Configure observation function
    List<String> sensors = new ArrayList<>();
    sensors.add("a");
    sensors.add("r");
    List<Grid.Key> keys = new ArrayList<>();
    keys.add(new Grid.Key(0, 0));
    keys.add(new Grid.Key(0, 2));
    keys.add(new Grid.Key(3, 0));
    keys.add(new Grid.Key(3, 2));
    ObservationFunction observationFunction = new ObservationFunction(sensors, keys);

    System.out.println("Observation function initialized with " + sensors.size() + " sensors, " + keys.size() + " keys and " + 1 + " steps => " + observationFunction.getOutputDimension() + " dimensions");
    System.out.println("Used sensors: " + sensors);

    // Configure reward function
    final int xReward = 1;
    final int yReward = 1;
    RewardFunction rewardFunction = new RewardFunction(xReward, yReward);

    System.out.println("Reward function initialized with x : " + xReward + " and y : " + yReward);

    // Configure input converter
    RLUtils.InputConverter inputConverter = new RLUtils.InputConverter(observationFunction.getOutputDimension(),
        nInputPartitions
    );

    // Configure output converter
    RLUtils.OutputConverter outputConverter = new RLUtils.OutputConverter(7, nOutputPartitions, maxAppliedForce);

    // Configure agent
    int stateDim = inputConverter.getMaxOutput() + 1;
    int actionDim = outputConverter.getMaxInput() + 1;
    ContinuousRL agent = new TabularSARSALambda(discountFactor,
        stateDim,
        actionDim,
        meanQ,
        stdQ,
        seed,
        inputConverter,
        outputConverter,
        lambda
    );

    // Configure the control function
    Set<Set<Grid.Key>> clusteredKeys = PoseUtils.computeClusteredByPositionPoses(morphology, 7, 42);
    List<List<Grid.Key>> keysList = new ArrayList<>();
    for (Set<Grid.Key> clusteredKey : clusteredKeys) {
      keysList.add(new ArrayList<>(clusteredKey.stream().toList()));
    }
    ControlFunction controlFunction = new ControlFunction(keysList, 4, 3);

    // Configure robot
    final double controllerFreq = 0.45;
    final RLController controller = new RLController(observationFunction, rewardFunction, agent, controlFunction);
    final StepController stepController = new StepController(controller, controllerFreq);
    final Robot robot = new Robot(stepController, body);

    System.out.println("Robot initialized");

    // Launch learning
    final int episodes = 1;
    final double episodeLength = 75d;
    final double[] pos = new double[episodes];

    for (int i = 0; i < episodes; i++) {
      //System.out.println("Episode " + i + " of " + episodes + " started");
      Predicate<Map<Double, Outcome.Observation>> earlyStopping = makeStoppingCriterion(episodeLength);
      Locomotion locomotion = new Locomotion(earlyStopping, createTerrain(), 2500d, new Settings());
      GridOnlineViewer.run(
          locomotion,
          Grid.create(1, 1, new NamedValue<>("RL Robot", robot)),
          Drawers::basicWithMiniWorldAndRL
      );
      System.exit(0);
      //RLControllerListener listener = new RLControllerListener();
      Outcome outcome = locomotion.apply(robot);
      pos[i] = outcome.getDistance();
      System.out.println(outcome.getDistance());
      //System.out.println("Episode length: " + outcome.getTime() + " seconds - PositionX: " + outcome.getDistance() + " m");
      //File file = new File(RESULTS_PATH + "biped-SARSA-episode-" + i + ".csv");
      //listener.toFile(file);
    }

    // write pos array as a csv file
    File file = new File(RESULTS_PATH + "biped-SARSA-pos.csv");

    try (PrintWriter out = new PrintWriter(file)) {
      out.println("distance");
      for (int i = 0; i < episodes; i++) {
        out.println(pos[i]);
      }
    }

    // Test the agent
    System.out.println("Testing the agent");
    final double testEpisodeLength = 60d;
    Locomotion locomotion = new Locomotion(testEpisodeLength, createTerrain(), 2500d, new Settings());
    GridFileWriter.save(locomotion,
        robot,
        600,
        400,
        0,
        20,
        VideoUtils.EncoderFacility.JCODEC,
        new File(RESULTS_PATH + "satsa-test.mp4")
    );

    System.out.println("Test finished");
  }

  public static void main(String[] args) throws IOException {
    runBipedTabularSARSA();
  }
}
