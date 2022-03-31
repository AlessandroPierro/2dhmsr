package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

public class TabularQLearning extends AbstractQTableAgent {

  public TabularQLearning(double discountFactor, int seed, int stateDim, int actionDim) {
    super(discountFactor, seed, stateDim, actionDim);
  }

  public TabularQLearning(
      double discountFactor, int seed, int stateDim, int actionDim, double[][] qTable
  ) {
    super(discountFactor, seed, stateDim, actionDim, qTable);
  }

  @Override
  protected void updateQTable(int newState, double reward, int newAction) {
    double delta = reward + discountFactor * getMaxQ(newState) - qTable[previousState][previousAction];
    qTable[previousState][previousAction] += learningRate * delta;
  }
}
