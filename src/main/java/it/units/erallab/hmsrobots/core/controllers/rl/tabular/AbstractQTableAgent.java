package it.units.erallab.hmsrobots.core.controllers.rl.tabular;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.rl.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.RLUtils;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;

import java.io.Serializable;
import java.util.Random;
import java.util.random.RandomGenerator;

public abstract class AbstractQTableAgent extends ContinuousRL implements Snapshottable, Serializable {
  @JsonProperty
  protected final double discountFactor;
  @JsonProperty
  protected final int stateDim;
  @JsonProperty
  protected final int actionDim;
  @JsonProperty
  protected double[][] qTable;
  @JsonProperty
  protected final int seed;
  @JsonProperty
  protected int episodeNumber;
  @JsonProperty
  protected transient RLUtils.InputConverter inputConverter;
  @JsonProperty
  protected transient RLUtils.OutputConverter outputConverter;

  // Internal variables
  protected transient RandomGenerator random;
  protected transient int previousState;
  protected transient int previousAction;
  protected transient int previousMaxAction;
  protected transient double previousReward;
  protected transient double epsilon;
  protected transient double learningRate;
  protected transient boolean learn = true;

  public AbstractQTableAgent(
      double discountFactor,
      int stateDim,
      int actionDim,
      double meanQ,
      double stdQ,
      int seed,
      RLUtils.InputConverter inputConverter,
      RLUtils.OutputConverter outputConverter
  ) {
    this.discountFactor = discountFactor;
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.qTable = new double[stateDim][actionDim];
    this.seed = seed;
    this.episodeNumber = 0;

    this.random = new Random(seed);
    for (int i = 0; i < stateDim; i++) {
      for (int j = 0; j < actionDim; j++) {
        qTable[i][j] = random.nextGaussian() * stdQ + meanQ;
      }
    }

    this.inputConverter = inputConverter;
    this.outputConverter = outputConverter;
    this.previousState = Integer.MIN_VALUE;
    this.previousAction = Integer.MIN_VALUE;
    this.previousReward = Double.MIN_VALUE;
  }

  public AbstractQTableAgent(
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("stateDim") int stateDim,
      @JsonProperty("actionDim") int actionDim,
      @JsonProperty("qTable") double[][] qTable,
      @JsonProperty("seed") int seed,
      @JsonProperty("episodeNumber") int episodeNumber,
      @JsonProperty("inputConverter") RLUtils.InputConverter inputConverter,
      @JsonProperty("outputConverter") RLUtils.OutputConverter outputConverter
  ) {
    this.discountFactor = discountFactor;
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.qTable = qTable;
    this.seed = seed;
    this.episodeNumber = episodeNumber;
    this.inputConverter = inputConverter;
    this.outputConverter = outputConverter;
  }

  @Override
  public double[] apply(double[] newState, Double reward) {
    int state = inputConverter.apply(newState);
    if (random == null) {
      random = new Random(seed);
    }
    epsilon = explorationRateSchedule();
    learningRate = learningRateSchedule();
    int action;
    int maxAction = getMaxAction(state);
    if (random.nextDouble() < epsilon) {
      action = random.nextInt(actionDim);
    } else {
      action = maxAction;
    }
    if (previousAction != Integer.MIN_VALUE && learn) {
      updateQTable(state, reward, action);
    }
    previousState = state;
    previousAction = action;
    previousMaxAction = maxAction;
    previousReward = reward;
    return outputConverter.apply(action);
  }

  // TODO : Finetune exploration rate scheduling
  private double explorationRateSchedule() {
    return Math.max(Math.pow(1d / (1d + episodeNumber), 0.8), 0.025);
  }

  // TODO : Finetume learning rate scheduling
  private double learningRateSchedule() {
    return 0.001;
  }

  public int getInputDimension() {
    return stateDim;
  }

  public int getOutputDimension() {
    return actionDim;
  }

  protected int getMaxAction(int state) {
    int maxAction = 0;
    for (int action = 1; action < actionDim; action++) {
      maxAction = qTable[state][action] > qTable[state][maxAction] ? action : maxAction;
    }
    return maxAction;
  }

  protected double getMaxQ(int state) {
    return qTable[state][getMaxAction(state)];
  }

  @Override
  public Snapshot getSnapshot() {
    // TODO : Implement snapshot / qtable state
    return new Snapshot(null, this.getClass());
  }

  @Override
  public void reset() {
    previousState = Integer.MIN_VALUE;
    previousAction = Integer.MIN_VALUE;
    previousReward = Double.MIN_VALUE;
    episodeNumber++;
  }

  protected void updateQTable(int newState, double reward, int action) {
  }

  protected int selectEpsGreedyAction(int state) {
    previousMaxAction = getMaxAction(state);
    return random.nextDouble() < epsilon ? random.nextInt(actionDim) : previousMaxAction;
  }

  public int getEpisodeNumber() {
    return episodeNumber;
  }

}
