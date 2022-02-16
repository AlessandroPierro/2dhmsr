package it.units.erallab.hmsrobots.core.snapshots;

import it.units.erallab.hmsrobots.core.snapshots.Snapshot;

public class RLControllerState {
  private final double reward;
  private final double[] observation;
  private final double[] action;
  private final Snapshot rlSnapshot;

  public RLControllerState(
      double reward,
      double[] observation,
      double[] action,
      Snapshot rlSnapshot
  ) {
    this.reward = reward;
    this.observation = observation;
    this.action = action;
    this.rlSnapshot = rlSnapshot;
  }

  public double[] getAction() {
    return action;
  }

  public double[] getObservation() {
    return observation;
  }

  public double getReward() {
    return reward;
  }

  public Snapshot getRlSnapshot() {
    return rlSnapshot;
  }
}
