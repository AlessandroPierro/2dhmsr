package it.units.erallab.hmsrobots.core.snapshots;

public class RLControllerState {
    private final double reward;
    private final double[] observation;
    private final double[] action;
    private final double velocityX;
    private final double velocityY;

    public RLControllerState(double reward, double[] observation, double[] action, double velocityX, double velocityY) {
        this.reward = reward;
        this.observation = observation;
        this.action = action;
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


    public double getVelocityX() {
        return velocityX;
    }

    public double getVelocityY() {
        return velocityY;
    }
}
