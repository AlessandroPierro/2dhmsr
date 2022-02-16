package it.units.erallab.hmsrobots.core.controllers.rl;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Supplier;
import java.util.random.RandomGenerator;

public abstract class AbstractQTableAgent implements DiscreteRL, Serializable {
  @JsonProperty
  protected final int inputDimension;
  @JsonProperty
  protected final int outputDimension;

  @JsonProperty
  protected final double[][] qTable;
  @JsonProperty
  protected final boolean episodic;
  @JsonProperty
  protected final double learningRateDecay;
  @JsonProperty
  protected final double explorationRateDecay;
  @JsonProperty
  protected final double discountFactor;
  @JsonProperty
  private final int seed;
  @JsonProperty
  protected double learningRate;
  @JsonProperty
  protected double explorationRate;
  protected final RandomGenerator random;
  protected boolean initialized = false;
  protected int previousState;
  protected int action;

  public AbstractQTableAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor, int seed,
      Supplier<Double> initializer,
      int inputDimension,
      int outputDimension,
      boolean episodic
  ) {
    this.learningRate = learningRate;
    this.explorationRate = explorationRate;
    this.learningRateDecay = learningRateDecay;
    this.explorationRateDecay = explorationRateDecay;
    this.discountFactor = discountFactor;
    this.seed = seed;
    this.random = new Random(seed);
    this.inputDimension = inputDimension;
    this.outputDimension = outputDimension;
    this.qTable = new double[inputDimension][outputDimension];

    for (int i = 0; i < inputDimension; i++) {
      for (int j = 0; j < outputDimension; j++) {
        qTable[i][j] = initializer.get();
      }
    }

    this.episodic = episodic;
  }


  public AbstractQTableAgent(
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
    this.learningRate = learningRate;
    this.explorationRate = explorationRate;
    this.learningRateDecay = learningRateDecay;
    this.explorationRateDecay = explorationRateDecay;
    this.discountFactor = discountFactor;
    this.seed = seed;
    this.random = new Random(seed);
    this.inputDimension = inputDimension;
    this.outputDimension = outputDimension;
    this.qTable = copyOf(qTable);
    this.episodic = episodic;
  }

  private static double[][] copyOf(double[][] o) {
    double[][] c = new double[o.length][];
    for (int i = 0; i < o.length; i++) {
      c[i] = Arrays.copyOf(o[i], o[i].length);
    }
    return c;
  }

  @Override
  public int apply(double t, int input, double r) {
    if (initialized) {
      updateQTable(previousState, action, input, r);
    } else {
      initialized = true;
    }

    action = random.nextDouble() < explorationRate ? random.nextInt(outputDimension) : getMaxAction(input);
    previousState = input;

    if (!episodic) {
      learningRate = learningRate * learningRateDecay;
      explorationRate = explorationRate * explorationRateDecay;
    }

    return action;
  }

  public void stopExploration() {
    this.explorationRate = 0.0;
  }

  public void setExplorationRate(double explorationRate) {
    this.explorationRate = explorationRate;
  }

  public int getInputDimension() {
    return inputDimension;
  }

  public int getOutputDimension() {
    return outputDimension;
  }

  protected int getMaxAction(int state) {
    double maxQ = Double.NEGATIVE_INFINITY;
    int maxAction = 0;
    for (int action = 0; action < outputDimension; action++) {
      if (qTable[state][action] > maxQ) {
        maxQ = qTable[state][action];
        maxAction = action;
      }
    }
    return maxAction;
  }

  protected double getMaxQ(int state) {
    double maxQ = Double.NEGATIVE_INFINITY;
    for (int action = 0; action < outputDimension; action++) {
      maxQ = Math.max(maxQ, qTable[state][action]);
    }
    return maxQ;
  }

  @Override
  public Snapshot getSnapshot() {
    QTableAgentState content = new QTableAgentState(
        qTable,
        inputDimension,
        outputDimension,
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor
    );
    return new Snapshot(content, this.getClass());
  }

  @Override
  public void reset() {
    initialized = false;
    learningRate = learningRate * learningRateDecay;
    explorationRate = explorationRate * explorationRateDecay;
  }

  protected void updateQTable(int previousState, int action, int newState, double r) {
  }
}
