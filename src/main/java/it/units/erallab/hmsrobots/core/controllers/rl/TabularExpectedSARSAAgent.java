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
      boolean episodic,
      int stateSpaceDimension,
      int actionSpaceDimension
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
        episodic,
        stateSpaceDimension,
        actionSpaceDimension
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
      @JsonProperty("seed") int seed,
      @JsonProperty("stateSpaceDimension") int stateSpaceDimension,
      @JsonProperty("actionSpaceDimension") int actionSpaceDimension
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
        seed,
        stateSpaceDimension,
        actionSpaceDimension
    );
  }

  @Override
  protected void updateQTable(int previousState, int action, int newState, double r) {
    double q = qTable[previousState][action];
    int maxQAction = getMaxAction(newState);
    double expectedSARSA = 0.0;
    for (int possibleAction = 0; possibleAction < actionSpaceDimension; possibleAction++) {
      expectedSARSA += qTable[newState][possibleAction] * (explorationRate / actionSpaceDimension + (possibleAction == maxQAction ? 1-explorationRate : 0.0));
    }
    qTable[previousState][action] = q + learningRate * (r + discountFactor * expectedSARSA - q);
  }
}
