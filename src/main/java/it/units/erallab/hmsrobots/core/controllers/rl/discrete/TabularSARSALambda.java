package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import java.util.Arrays;

public class TabularSARSALambda extends AbstractQTableAgent {

  private final double lambda;
  private final double[][] eTraces;

  public TabularSARSALambda(
      double discountFactor, double lambda, int stateDim, int actionDim, double meanQ, double stdQ, int seed
  ) {
    super(discountFactor, stateDim, actionDim, meanQ, stdQ, seed);
    this.lambda = lambda;
    this.eTraces = new double[stateDim][actionDim];
    for (int i = 0; i < stateDim; i++) {
      Arrays.fill(eTraces[i], 0d);
    }
  }

  @Override
  public void reset() {
    super.reset();
    for (int i = 0; i < stateDim; i++) {
      Arrays.fill(eTraces[i], 0d);
    }
  }

  @Override
  protected void updateQTable(int newState, double reward, int newAction) {
    double delta = reward + discountFactor * qTable[newState][newAction] - qTable[previousState][previousAction];
    eTraces[previousState][previousAction] += 1;
    for (int i = 0; i < stateDim; i++) {
      for (int j = 0; j < actionDim; j++) {
        qTable[i][j] += learningRate * delta * eTraces[i][j];
        eTraces[i][j] *= discountFactor * lambda;
      }
    }
  }
}
