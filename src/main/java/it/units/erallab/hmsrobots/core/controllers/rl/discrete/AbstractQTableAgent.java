package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.random.RandomGenerator;

public abstract class AbstractQTableAgent implements DiscreteRL, Serializable {
  @JsonProperty
  protected final double learningRateDecay;
  @JsonProperty
  protected final double explorationRateDecay;
  @JsonProperty
  protected final double discountFactor;
  @JsonProperty
  protected final int stateSpaceDimension;
  @JsonProperty
  protected final int actionSpaceDimension;
  protected final RandomGenerator random;
  @JsonProperty
  private final int seed;
  @JsonProperty
  protected double[][] qTableA;

  protected boolean learn = true;
  protected boolean explore = true;
  @JsonProperty
  protected int[] nLearningVisits;
  @JsonProperty
  protected int[] nExplorationVisits;
  @JsonProperty
  protected double[] learningRates;
  @JsonProperty
  protected double[] explorationRates;

  protected boolean initialized = false;
  protected int previousState;
  protected int action;

  public AbstractQTableAgent(
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor,
      int seed,
      int stateSpaceDimension,
      int actionSpaceDimension
  ) {
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
        qTableA[i][j] = 0d;
      }
    }

    nLearningVisits = new int[stateSpaceDimension];
    nExplorationVisits = new int[stateSpaceDimension];
    learningRates = new double[stateSpaceDimension];
    Arrays.fill(learningRates, 1d);
    explorationRates = new double[stateSpaceDimension];
    Arrays.fill(explorationRates, 1d);
  }


  public AbstractQTableAgent(
      @JsonProperty("learningRateDecay") double learningRateDecay,
      @JsonProperty("explorationRateDecay") double explorationRateDecay,
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("qTable") double[][] qTable,
      @JsonProperty("seed") int seed,
      @JsonProperty("stateSpaceDimension") int stateSpaceDimension,
      @JsonProperty("actionSpaceDimension") int actionSpaceDimension,
      @JsonProperty("nLearningVisits") int[] nLearningVisits,
      @JsonProperty("nExplorationVisits") int[] nExplorationVisits,
      @JsonProperty("learningRates") double[] learningRates,
      @JsonProperty("explorationRates") double[] explorationRates
  ) {
    this(learningRateDecay, explorationRateDecay, discountFactor, seed, stateSpaceDimension, actionSpaceDimension);
    this.qTableA = Arrays.copyOf(qTable, qTable.length);
    this.nLearningVisits = Arrays.copyOf(nLearningVisits, nLearningVisits.length);
    this.nExplorationVisits = Arrays.copyOf(nExplorationVisits, nExplorationVisits.length);
    this.learningRates = Arrays.copyOf(learningRates, learningRates.length);
    this.explorationRates = Arrays.copyOf(explorationRates, explorationRates.length);
  }

  @Override
  public int apply(double t, int newState, double reward) {
    if (initialized && learn) {
      updateQTable(previousState, action, reward, newState, qTableA);
      updateLearningRates(newState);
    }
    initialized = true;
    if (explore) {
      updateExplorationRates(newState);
      action = random.nextDouble() < explorationRates[newState] ? random.nextInt(actionSpaceDimension) : getMaxAction(
          newState,
          qTableA
      );
    } else {
      action = getMaxAction(newState, qTableA);
    }
    previousState = newState;
    return action;
  }

  public void freezeExploration() {
    explore = false;
  }

  public void freezeLearning() {
    learn = false;
  }

  public int getInputDimension() {
    return stateSpaceDimension;
  }

  public int getOutputDimension() {
    return actionSpaceDimension;
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
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        previousState
    );
    return new Snapshot(content, this.getClass());
  }

  @Override
  public void reset() {
    initialized = false;
  }

  public void unfreezeExploration() {
    explore = true;
  }

  public void unfreezeLearning() {
    learn = true;
  }

  protected void updateExplorationRates(int newState) {
    nExplorationVisits[newState]++;
    explorationRates[newState] = Math.max(1 / Math.pow(nExplorationVisits[newState], explorationRateDecay), 0.05);
  }

  protected void updateLearningRates(int newState) {
    nLearningVisits[newState]++;
    learningRates[newState] = 1 / Math.pow(nLearningVisits[newState], learningRateDecay);
  }

  protected void updateQTable(int previousState, int action, double reward, int newState, double[][] qTable) {
  }
}
