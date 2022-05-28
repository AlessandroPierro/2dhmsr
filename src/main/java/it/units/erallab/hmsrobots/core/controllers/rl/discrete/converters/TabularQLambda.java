package it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters;

import it.units.erallab.hmsrobots.core.controllers.rl.discrete.TabularSARSALambda;

public class TabularQLambda extends TabularSARSALambda {
  public TabularQLambda(
      double discountFactor,
      double lambda,
      int stateDim,
      int actionDim,
      double meanQ,
      double stdQ,
      int seed
  ) {
    super(discountFactor, lambda, stateDim, actionDim, meanQ, stdQ, seed);
  }

  @Override
  protected void updateQTable(int newState, double reward, int newAction) {
    int maxAction = getMaxAction(newState);
    double delta = reward + discountFactor * qTable.getEntry(newState, maxAction) - qTable.getEntry(
        previousState,
        previousAction
    );
    eTraces.addToEntry(previousState, previousAction, 1);
    qTable = qTable.add(eTraces.scalarMultiply(lambda).scalarMultiply(delta));
    eTraces = eTraces.scalarMultiply(lambda).scalarMultiply(discountFactor);
    if (maxAction != newAction) {
      double temp = qTable.getEntry(previousState, previousAction) * lambda * discountFactor;
      eTraces = eTraces.scalarMultiply(0d);
      eTraces.setEntry(previousState, previousAction, temp);
    }
  }
}
