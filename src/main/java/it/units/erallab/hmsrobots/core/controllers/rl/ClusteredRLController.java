package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.core.snapshots.RLControllerState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;

import java.util.*;
import java.util.function.DoubleBinaryOperator;
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
      List<List<Grid.Key>> clusters,
      LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map,
      ContinuousRL rl,
      ToDoubleFunction<Grid<Voxel>> rewardFunction
  ) {
    this.rl = rl;
    this.rewardFunction = rewardFunction;
    this.observationFunction = new ClusteredObservationFunction(map);
    this.controlFunction = new ClusteredControlFunction(clusters);
  }

  @Override
  public Grid<Double> computeControlSignals(
      double t, Grid<Voxel> voxels
  ) {
    System.out.println("Computing control signals...");
    reward = rewardFunction.applyAsDouble(voxels);
    System.out.println("Reward: " + reward);
    observation = observationFunction.apply(t, voxels);
    System.out.println("Observation: " + Arrays.toString(observation));
    action = rl.apply(t, observation, reward);
    System.out.println("Action: " + Arrays.toString(action));
    controlSignals = controlFunction.apply(action);
    System.out.println("Control signals: " + controlSignals);
    return controlSignals;
  }

  public int getReadingsDimension() {
    return observationFunction.getOutputDimension();
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