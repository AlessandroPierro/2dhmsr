package it.units.erallab.hmsrobots.core.snapshots;

public class RLControllerState {
  private final double reward;
  private final double[] observation;
  private final double[] action;
  private final double velocity;

  public RLControllerState(
      double reward,
      double[] observation,
      double[] action,
      double velocity
  ) {
    this.reward = reward;
    this.observation = observation;
    this.action = action;
    this.velocity = velocity;
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

  public double getVelocity() {
    return velocity;
  }

}
