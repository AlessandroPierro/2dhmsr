package it.units.erallab.hmsrobots.core.controllers.rl;

import com.google.common.util.concurrent.AtomicDouble;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.CompositeSensor;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.core.sensors.Velocity;
import it.units.erallab.hmsrobots.core.snapshots.RLControllerState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.ToDoubleFunction;

public class RLController extends AbstractController implements Snapshottable {

  private final ToDoubleFunction<Grid<Voxel>> rewardFunction;
  private final BiFunction<Double, Grid<Voxel>, double[]> observationFunction;
  private final ContinuousRL rl;
  private final List<List<Grid.Key>> clusters;
  private Grid<Double> output;
  private boolean initialized = false;

  private double[] observation;
  private double reward;
  private double[] action;

  private int totalSteps = 0;
  private double maxSpeed = Double.NEGATIVE_INFINITY;
  private double minSpeed = Double.POSITIVE_INFINITY;
  private double averageSpeed = 0.0;
  private double maxReward = Double.NEGATIVE_INFINITY;
  private double minReward = Double.POSITIVE_INFINITY;
  private double averageReward = 0.0;

  public RLController(
      ToDoubleFunction<Grid<Voxel>> rewardFunction,
      BiFunction<Double, Grid<Voxel>, double[]> observationFunction,
      ContinuousRL rl,
      List<List<Grid.Key>> clusters
  ) {
    this.rewardFunction = rewardFunction;
    this.observationFunction = observationFunction;
    this.rl = rl;
    this.clusters = clusters;
  }

  @Override
  public Grid<Double> computeControlSignals(
      double t, Grid<Voxel> voxels
  ) {
    if (!initialized) {
      output = Grid.create(voxels.getW(), voxels.getH());
      initialized = true;
    }
    observation = observationFunction.apply(t, voxels);
    if (observation.length != rl.getInputDimension()) {
      throw new IllegalArgumentException(String.format(
          "Observation dim different than expected: %d vs %d",
          observation.length,
          rl.getInputDimension()
      ));
    }

    totalSteps += 1;

    /* Track speed
    int counter = 0;
    double speed = 0.0;
    for (List<Grid.Key> cluster : clusters) {
      for (Grid.Key key : cluster) {
        // TODO : Improve this
        if (voxels.get(key.x(), key.y()).getSensors().get(1) instanceof CompositeSensor) {
          speed += ((CompositeSensor) voxels.get(key.x(), key.y()).getSensors().get(1)).getSensor()
              .getReadings()[0];
          ++counter;
        }
      }
    }
    averageSpeed = (averageSpeed * (totalSteps - 1) + speed.get() / (double) n.get()) / totalSteps;
    maxSpeed = Math.max(maxSpeed, speed.get() / (double) n.get());
    minSpeed = Math.min(minSpeed, speed.get() / (double) n.get());
*/
    // Track reward
    reward = rewardFunction.applyAsDouble(voxels);
    averageReward = (averageReward * (totalSteps - 1) + reward) / totalSteps;
    maxReward = Math.max(reward, maxReward);
    minReward = Math.min(reward, minReward);

    action = rl.apply(t, observation, reward);
    int nOfVoxels = (int) voxels.stream().map(Grid.Entry::value).filter(Objects::nonNull).count();
    if (action.length != nOfVoxels) {
      throw new IllegalArgumentException(String.format(
          "Action dim different than expected: %d vs %d",
          action.length,
          nOfVoxels
      ));
    }
    int c = 0;
    for (List<Grid.Key> cluster : clusters) {
      for (Grid.Key key : cluster) {
        output.set(key.x(), key.y(), action[c]);
        c++;
      }
    }
    return output;
  }

  public double getAverageReward() {
    return averageReward;
  }

  public double getMaxReward() {
    return maxReward;
  }

  public double getMinReward() {
    return minReward;
  }

  public double getAverageSpeed() {
    return averageSpeed;
  }

  public double getMaxSpeed() {
    return maxSpeed;
  }

  public double getMinSpeed() {
    return minSpeed;
  }

  @Override
  public Snapshot getSnapshot() {
    Snapshot snapshot = new Snapshot(
        new RLControllerState(reward, observation, action),
        getClass()
    );
    snapshot.getChildren().add(rl.getSnapshot());
    return snapshot;
  }

  @Override
  public void reset() {
    rl.reset();
    averageReward = 0.0;
    totalSteps = 0;
    maxReward = Double.NEGATIVE_INFINITY;
    minReward = Double.POSITIVE_INFINITY;
    if (rewardFunction instanceof Resettable r) {
      r.reset();
    }
  }

}
