package it.units.erallab.hmsrobots.core.controllers.dqn;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.Arrays;

public class RLController extends AbstractController implements Snapshottable {

  @JsonProperty
  protected final ObservationFunction observationFunction;
  @JsonProperty
  protected final RewardFunction rewardFunction;
  @JsonProperty
  protected final ContinuousRL rlAgent;
  @JsonProperty
  protected final ControlFunction controlFunction;

  protected transient double[] observation;
  protected transient double reward;
  protected transient double[] action;
  protected transient Grid<Double> control;

  public RLController(
      @JsonProperty("observationFunction") ObservationFunction observationFunction,
      @JsonProperty("rewardFunction") RewardFunction rewardFunction,
      @JsonProperty("rlAgent") ContinuousRL rlAgent,
      @JsonProperty("controlFunction") ControlFunction controlFunction
  ) {
    this.observationFunction = observationFunction;
    this.rewardFunction = rewardFunction;
    this.rlAgent = rlAgent;
    this.controlFunction = controlFunction;
  }

  @Override
  public Grid<Double> computeControlSignals(double t, Grid<Voxel> voxels) {
    observation = observationFunction.apply(t, voxels);
    //System.out.println("observation: " + Arrays.toString(observation));
    reward = rewardFunction.apply(t, voxels);
    //System.out.println("reward: " + reward);
    action = rlAgent.apply(observation, reward);
    //System.out.println("action: " + Arrays.toString(action));
    control = controlFunction.apply(t, action);
    //System.out.println("control: " + control);
    return control;
  }

  @Override
  public Snapshot getSnapshot() {
    RLControllerState state = new RLControllerState(observation.clone(), reward, action.clone(), control);
    return new Snapshot(state, getClass());
  }

  @Override
  public void reset() {
    observationFunction.reset();
    rewardFunction.reset();
    rlAgent.reset();
    controlFunction.reset();
  }
}
