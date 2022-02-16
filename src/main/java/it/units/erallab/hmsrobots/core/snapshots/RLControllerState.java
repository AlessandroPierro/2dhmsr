package it.units.erallab.hmsrobots.core.snapshots;

public class RLControllerState {
  private final double reward;
  private final double[] observation;
  private final double[] action;

  public RLControllerState(
      double reward,
      double[] observation,
      double[] action
  ) {
    this.reward = reward;
    this.observation = observation;
    this.action = action;
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

}
