package it.units.erallab.hmsrobots.core.controllers.rl.continuous;

import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryOutputConverter;
import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;

import java.util.Arrays;
import java.util.Random;

public class RBFSARSALambda implements ContinuousRL, Snapshottable {

  private final double discountFactor;
  private final double lambda;
  private final int stateDim;
  private final int actionDim;
  private final Random random;
  private final DiscreteRL.OutputConverter outputConverter;
  private final int nFeatures;
  private final double[][] centroids;
  private double[][] weights;
  private final double[] eTraces;
  private double epsilon;
  private double learningRate;
  private boolean learn = true;
  private int episodeNumber = 0;
  private double sigma = 0.15;

  private int previousAction = Integer.MIN_VALUE;
  private double[] previousState = null;
  private double[] previousFeatures = null;
  private double previousQ;

  public RBFSARSALambda(
      double lambda, double discountFactor, int stateDim, int actionDim, double initMean, double initStd, int seed
  ) {
    this.discountFactor = discountFactor;
    this.lambda = lambda;
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.nFeatures = (int) Math.pow(4, stateDim) + 1;
    this.weights = new double[actionDim][nFeatures];
    this.eTraces = new double[nFeatures];
    Arrays.fill(eTraces, 0d);
    this.random = new java.util.Random(seed);
    for (int i = 0; i < actionDim; i++) {
      for (int j = 0; j < nFeatures; j++) {
        weights[i][j] = random.nextGaussian() * initStd + initMean;
      }
    }
    this.outputConverter = new BinaryOutputConverter((int) (Math.log(actionDim) / Math.log(2)), 1);

    this.centroids = new double[nFeatures][stateDim];
    // centroids as all possible permutations of {0, 0.25, 0.75, 1} for each dimension
    double[] values = new double[]{0d, 0.25, 0.75, 1d};
    for (int i = 0; i < nFeatures; i++) {
      for (int j = 0; j < stateDim; j++) {
        centroids[i][j] = values[i / (int) Math.pow(3, j) % 4];
      }
    }
  }

  public RBFSARSALambda(
      double lambda, double discountFactor, int stateDim, int actionDim, double initMean, double initStd, int seed, double[][] weights
  ) {
    this(lambda, discountFactor, stateDim, actionDim, initMean, initStd, seed);
    this.weights = Arrays.stream(weights).map(double[]::clone).toArray(double[][]::new);
  }

  @Override
  public double[] apply(double t, double[] newState, double reward) {
    epsilon = explorationRateSchedule(t);
    learningRate = learningRateSchedule(t);
    System.out.println("epsilon: " + epsilon);
    System.out.println("learning rate: " + learningRate);
    double[] features = computeFeatures(newState);
    int action = selectEpsGreedyAction(newState);
    double newQ = computeQvalues(features, action);
    if (previousAction != Integer.MIN_VALUE && learn) {
      double delta = reward + discountFactor * newQ - previousQ;
      double norm = 0d;
      for (int i = 0; i < nFeatures; i++) {
        eTraces[i] = previousFeatures[i] + lambda * discountFactor * eTraces[i];
        weights[action][i] += learningRate * delta * eTraces[i];
        norm += Math.pow(weights[action][i], 2);
      }
      norm = Math.sqrt(norm);
      if (norm > 1) {
        for (int i = 0; i < nFeatures; i++) {
          weights[action][i] /= norm;
        }
      }
    }
    previousState = newState;
    previousFeatures = features;
    previousAction = action;
    previousQ = newQ;
    return outputConverter.apply(action);
  }

  private double[] computeFeatures(double[] state) {
    double[] features = new double[nFeatures];
    for (int i = 0; i < nFeatures - 1; i++) {
      double distance = 0;
      for (int j = 0; j < stateDim; j++) {
        distance += Math.pow(state[j] - centroids[i][j], 2);
      }
      distance = Math.sqrt(distance);
      features[i] = Math.exp(-distance / (2 * Math.pow(sigma, 2)));
    }
    features[nFeatures - 1] = 1;
    return features;
  }

  double computeQvalues(double[] features, int action) {
    double q = 0d;
    for (int i = 0; i < nFeatures; i++) {
      q += weights[action][i] * features[i];
    }
    return q;
  }

  private double explorationRateSchedule(double t) {
    return learn ? 1d / (1 + Math.pow(t / 100, 0.25)) : 0.05;
  }

  @Override
  public int getInputDimension() {
    return stateDim;
  }

  @Override
  public int getOutputDimension() {
    return (int) (Math.log(actionDim) / Math.log(2));
  }

  @Override
  public Snapshot getSnapshot() {
    return new Snapshot(
        new QTableAgentState(
            Arrays.stream(weights).map(double[]::clone).toArray(double[][]::new),
            actionDim,
            nFeatures,
            0d,
            0d,
            4
        ),
        this.getClass()
    );
  }

  private double learningRateSchedule(double t) {
    return 0.1;
  }

  @Override
  public void reset() {
    episodeNumber++;
    previousAction = Integer.MIN_VALUE;
    previousState = null;
    Arrays.fill(eTraces, 0d);
  }

  private int selectEpsGreedyAction(double[] state) {
    if (random.nextDouble() < epsilon) {
      return random.nextInt(actionDim);
    }
    double[] features = computeFeatures(state);
    double maxQ = Double.NEGATIVE_INFINITY;
    int maxAction = 0;
    for (int i = 0; i < actionDim; i++) {
      double q = computeQvalues(features, i);
      maxAction = q > maxQ ? i : maxAction;
      maxQ = Math.max(q, maxQ);
    }
    return maxAction;
  }

  public void startLearning() {
    learn = true;
  }

  public void stopLearning() {
    learn = false;
    System.out.print("weights: ");
    System.out.println(Arrays.deepToString(weights));
  }
}
