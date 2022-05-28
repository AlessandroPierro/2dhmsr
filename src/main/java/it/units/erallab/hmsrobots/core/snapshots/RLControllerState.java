package it.units.erallab.hmsrobots.core.snapshots;

public class RLControllerState {
  private final double reward;
  private final double[] observation;
  private final double[] action;
  private final double positionX;
  private final double positionY;
  private final double velocityX;
  private final double velocityY;

  public RLControllerState(
      double reward,
      double[] observation,
      double[] action,
      double positionX,
      double positionY,
      double velocityX,
      double velocityY
  ) {
    this.reward = reward;
    this.observation = observation;
    this.action = action;
    this.positionX = positionX;
    this.positionY = positionY;
    this.velocityX = velocityX;
    this.velocityY = velocityY;
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

  public double getPositionX() {
    return positionX;
  }

  public double getPositionY() {
    return positionY;
  }

  public double getVelocityX() {
    return velocityX;
  }

  public double getVelocityY() {
    return velocityY;
  }
}
