package it.units.erallab.hmsrobots.core.controllers.rl;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.function.Supplier;

public class TabularExpectedSARSAAgent extends AbstractQTableAgent {

  public TabularExpectedSARSAAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor,
      int seed,
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
        seed,
        initializer,
        inputDimension,
        outputDimension,
        episodic
    );
  }

  public TabularExpectedSARSAAgent(
      @JsonProperty("learningRate") double learningRate,
      @JsonProperty("explorationRate") double explorationRate,
      @JsonProperty("learningRateDecay") double learningRateDecay,
      @JsonProperty("explorationRateDecay") double explorationRateDecay,
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("inputDimension") int inputDimension,
      @JsonProperty("outputDimension") int outputDimension,
      @JsonProperty("episodic") boolean episodic,
      @JsonProperty("qTable") double[][] qTable,
      @JsonProperty("seed") int seed
  ) {
    super(
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        inputDimension,
        outputDimension,
        episodic,
        qTable,
        seed
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
