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
  protected final int stateSpaceDimension;
  @JsonProperty
  protected final int actionSpaceDimension;

  @JsonProperty
  protected final double[][] qTableA;
  @JsonProperty
  protected final double learningRateDecay;
  @JsonProperty
  protected final double explorationRateDecay;
  @JsonProperty
  protected final double discountFactor;
  protected final RandomGenerator random;
  @JsonProperty
  private final int seed;
  @JsonProperty
  protected double learningRate;
  @JsonProperty
  protected double explorationRate;
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
      int stateSpaceDimension,
      int actionSpaceDimension
  ) {
    this.learningRate = learningRate;
    this.explorationRate = explorationRate;
    this.learningRateDecay = learningRateDecay;
    this.explorationRateDecay = explorationRateDecay;
    this.discountFactor = discountFactor;
    this.seed = seed;
    this.random = new Random(seed);
    this.stateSpaceDimension = stateSpaceDimension;
    this.actionSpaceDimension = actionSpaceDimension;
    this.qTableA = new double[stateSpaceDimension][actionSpaceDimension];

    for (int i = 0; i < stateSpaceDimension; i++) {
      for (int j = 0; j < actionSpaceDimension; j++) {
        qTableA[i][j] = initializer.get();
      }
    }
  }


  public AbstractQTableAgent(
      @JsonProperty("learningRate") double learningRate,
      @JsonProperty("explorationRate") double explorationRate,
      @JsonProperty("learningRateDecay") double learningRateDecay,
      @JsonProperty("explorationRateDecay") double explorationRateDecay,
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("qTable") double[][] qTable,
      @JsonProperty("seed") int seed,
      @JsonProperty("stateSpaceDimension") int stateSpaceDimension,
      @JsonProperty("actionSpaceDimension") int actionSpaceDimension
  ) {
    this.learningRate = learningRate;
    this.explorationRate = explorationRate;
    this.learningRateDecay = learningRateDecay;
    this.explorationRateDecay = explorationRateDecay;
    this.discountFactor = discountFactor;
    this.seed = seed;
    this.random = new Random(seed);
    this.qTableA = copyOf(qTable);
    this.stateSpaceDimension = stateSpaceDimension;
    this.actionSpaceDimension = actionSpaceDimension;
  }

  private static double[][] copyOf(double[][] o) {
    double[][] c = new double[o.length][];
    for (int i = 0; i < o.length; i++) {
      c[i] = Arrays.copyOf(o[i], o[i].length);
    }
    return c;
  }

  @Override
  public int apply(double t, int newState, double reward) {
    if (initialized) {
      updateQTable(previousState, action, reward, newState, qTableA);
    } else {
      initialized = true;
    }

    action = random.nextDouble() < explorationRate ? random.nextInt(actionSpaceDimension) : getMaxAction(
        newState,
        qTableA
    );
    previousState = newState;
    learningRate = learningRate * learningRateDecay;
    explorationRate = explorationRate * explorationRateDecay;

    return action;
  }

  public double getExplorationRate() {
    return explorationRate;
  }

  public void setExplorationRate(double explorationRate) {
    this.explorationRate = explorationRate;
  }

  public int getInputDimension() {
    return stateSpaceDimension;
  }

  public int getOutputDimension() {
    return actionSpaceDimension;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  protected int getMaxAction(int state, double[][] qTable) {
    double maxQ = Double.NEGATIVE_INFINITY;
    int maxAction = 0;
    for (int action = 0; action < actionSpaceDimension; action++) {
      if (qTable[state][action] > maxQ) {
        maxQ = qTable[state][action];
        maxAction = action;
      }
    }
    return maxAction;
  }

  protected double getMaxQ(int state, double[][] qTable) {
    double maxQ = Double.NEGATIVE_INFINITY;
    for (int action = 0; action < actionSpaceDimension; action++) {
      maxQ = Math.max(maxQ, qTable[state][action]);
    }
    return maxQ;
  }

  @Override
  public Snapshot getSnapshot() {
    QTableAgentState content = new QTableAgentState(
        qTableA,
        stateSpaceDimension,
        actionSpaceDimension,
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
  }

  protected void updateQTable(int previousState, int action, double reward, int newState, double[][] qTable) {
  }
}
