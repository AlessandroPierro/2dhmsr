package it.units.erallab.hmsrobots.core.controllers.rl;

import java.util.function.Supplier;
import java.util.random.RandomGenerator;

public class TabularExpectedSARSAAgent extends AbstractQTableAgent {

  public TabularExpectedSARSAAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor,
      RandomGenerator random,
      Supplier<Double> initializer,
      int inputDimension,
      int outputDimension,
      boolean episodic
  ) {
    super(
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        random,
        initializer,
        inputDimension,
        outputDimension,
        episodic
    );
  }

  @Override
  protected void updateQTable(int previousState, int action, int newState, double r) {
    double q = qTable[previousState][action];
    int maxQAction = getMaxAction(newState);
    double expectedSARSA = 0.0;
    for (int possibleAction = 0; possibleAction < outputDimension; possibleAction++) {
      expectedSARSA += qTable[newState][possibleAction] * (possibleAction == maxQAction ? 0.9 : (0.1 / (outputDimension - 1)));
    }
    qTable[previousState][action] = q + learningRate * (r + discountFactor * expectedSARSA - q);
  }
}
