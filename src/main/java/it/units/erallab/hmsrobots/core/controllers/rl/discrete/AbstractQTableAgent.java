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
  protected double[] learningRates;
  @JsonProperty
  protected int[][] ucbVisits;
  protected boolean initialized = false;
  protected int previousState;
  protected int action;
  double c;

  public AbstractQTableAgent(
      double learningRateDecay,
      double discountFactor,
      double c,
      int seed,
      int stateSpaceDimension,
      int actionSpaceDimension
  ) {
    this.learningRateDecay = learningRateDecay;
    this.discountFactor = discountFactor;
    this.seed = seed;
    this.random = new Random(seed);
    this.stateSpaceDimension = stateSpaceDimension;
    this.actionSpaceDimension = actionSpaceDimension;
    this.qTableA = new double[stateSpaceDimension][actionSpaceDimension];
    for (int i = 0; i < stateSpaceDimension; i++) {
      for (int j = 0; j < actionSpaceDimension; j++) {
        qTableA[i][j] = 0;
      }
    }
    nLearningVisits = new int[stateSpaceDimension];
    Arrays.fill(nLearningVisits, 0);
    learningRates = new double[stateSpaceDimension];
    Arrays.fill(learningRates, 1d);
    ucbVisits = new int[stateSpaceDimension][actionSpaceDimension];
    for (int i = 0; i < stateSpaceDimension; i++) {
      for (int j = 0; j < actionSpaceDimension; j++) {
        ucbVisits[i][j] = 1;
      }
    }
    this.c = c;
  }


  public AbstractQTableAgent(
      @JsonProperty("learningRateDecay") double learningRateDecay,
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("qTable") double[][] qTable,
      @JsonProperty("seed") int seed,
      @JsonProperty("stateSpaceDimension") int stateSpaceDimension,
      @JsonProperty("actionSpaceDimension") int actionSpaceDimension,
      @JsonProperty("nLearningVisits") int[] nLearningVisits,
      @JsonProperty("learningRates") double[] learningRates
  ) {
    this(learningRateDecay, discountFactor, 4.0, seed, stateSpaceDimension, actionSpaceDimension);
    for (int i = 0; i < stateSpaceDimension; i++) {
      for (int j = 0; j < actionSpaceDimension; j++) {
        qTable[i][j] = 0d;
      }
    }
    this.nLearningVisits = Arrays.copyOf(nLearningVisits, nLearningVisits.length);
    this.learningRates = Arrays.copyOf(learningRates, learningRates.length);
  }

  @Override
  public int apply(double t, int newState, double reward) {
    if (initialized && learn) {
      updateQTable(previousState, action, reward, newState, qTableA);
      updateLearningRates(newState);
    }
    initialized = true;
    action = ucbActionSelection(t, newState);
    ucbVisits[newState][action] += 1;
    previousState = newState;
    return action;
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
        discountFactor,
        previousState
    );
    return new Snapshot(content, this.getClass());
  }

  @Override
  public void reset() {
    initialized = false;
  }

  protected int ucbActionSelection(double t, int state) {
    int ucbBestAction = 0;
    double ucbBestValue = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < actionSpaceDimension; i++) {
      double currentValue = this.qTableA[state][i] + this.c * Math.sqrt(Math.log(t) / ucbVisits[state][i]);
      if (currentValue > ucbBestValue) {
        ucbBestAction = i;
        ucbBestValue = currentValue;
      }
    }
    return ucbBestAction;
  }

  protected void updateLearningRates(int newState) {
    nLearningVisits[newState]++;
    learningRates[newState] = 1 / Math.pow(nLearningVisits[newState], learningRateDecay);
  }

  protected void updateQTable(int previousState, int action, double reward, int newState, double[][] qTable) {
  }
}
