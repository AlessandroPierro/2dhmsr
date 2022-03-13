package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.snapshots.RLControllerState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;

import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;

public class ClusteredRLController extends AbstractController implements Snapshottable {

  private final ToDoubleFunction<Grid<Voxel>> rewardFunction;
  private final ClusteredObservationFunction observationFunction;
  private final Function<double[], Grid<Double>> controlFunction;
  private ContinuousRL rl;
  private double[] observation;
  private double reward;
  private double[] action;
  private Grid<Double> controlSignals;

  public ClusteredRLController(
      String shape,
      String sensorConfig,
      ContinuousRL rl,
      ToDoubleFunction<Grid<Voxel>> rewardFunction
  ) {
    Grid<Boolean> testShape = RobotUtils.buildShape(shape);
    Set<Set<Grid.Key>> clustersSet = computeCardinalPoses(testShape);
    List<List<Grid.Key>> clusters = clustersSet.stream().map(s -> s.stream().toList()).toList();
    this.rl = rl;
    this.rewardFunction = rewardFunction;
    this.observationFunction = new ClusteredObservationFunction(clusters, sensorConfig);
    this.controlFunction = new ClusteredControlFunction(clusters);
  }

  @Override
  public Grid<Double> computeControlSignals(
      double t, Grid<Voxel> voxels
  ) {
    reward = rewardFunction.applyAsDouble(voxels);
    observation = observationFunction.apply(t, voxels);
    action = rl.apply(t, observation, reward);
    controlSignals = controlFunction.apply(action);
    return controlSignals;
  }

  public int getReadingsDimension() {
    return observationFunction.getnSensorReadings();
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
    if (rewardFunction instanceof Resettable r) {
      r.reset();
    }
  }

  public void setRL(ContinuousRL rl) {
    this.rl = rl;
  }
}