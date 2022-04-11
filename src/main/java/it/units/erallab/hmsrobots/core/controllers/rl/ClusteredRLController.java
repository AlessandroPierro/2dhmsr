package it.units.erallab.hmsrobots.core.controllers.rl;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.core.snapshots.RLControllerState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Grid;

import java.io.Serializable;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

public class ClusteredRLController extends AbstractController implements Snapshottable, Serializable {

  @JsonProperty
  private final ToDoubleFunction<Grid<Voxel>> rewardFunction;
  @JsonProperty
  private final ClusteredObservationFunction observationFunction;
  @JsonProperty
  private final Function<double[], Grid<Double>> controlFunction;
  @JsonProperty
  private ContinuousRL rl;

  private double[] observation;
  private double reward;
  private double[] action;

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

  public ClusteredRLController(
      @JsonProperty("rewardFunction") ToDoubleFunction<Grid<Voxel>> rewardFunction,
      @JsonProperty("observationFunction") ClusteredObservationFunction observationFunction,
      @JsonProperty("controlFunction") Function<double[], Grid<Double>> controlFunction,
      @JsonProperty("rl") ContinuousRL rl
  ) {
    this.rewardFunction = rewardFunction;
    this.observationFunction = observationFunction;
    this.controlFunction = controlFunction;
    this.rl = rl;
  }

  @Override
  public Grid<Double> computeControlSignals(
      double t, Grid<Voxel> voxels
  ) {
    reward = rewardFunction.applyAsDouble(voxels);
    observation = observationFunction.apply(t, voxels);
    action = rl.apply(t, observation, reward);
    return controlFunction.apply(action);
  }

  @Override
  public Snapshot getSnapshot() {
    Snapshot snapshot = new Snapshot(new RLControllerState(reward, observation, action), getClass());
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
}