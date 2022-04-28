package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

public class TabularQLearning extends AbstractQTableAgent {

  public TabularQLearning(double discountFactor, int stateDim, int actionDim, double meanQ, double stdQ, int seed) {
    super(discountFactor, stateDim, actionDim, meanQ, stdQ, seed);
  }

  @Override
  protected void updateQTable(int newState, double reward, int newAction) {
    double delta = reward + discountFactor * getMaxQ(newState) - qTable.getEntry(previousState, previousAction);
    qTable.addToEntry(previousState, previousAction, learningRate * delta);
  }
}
