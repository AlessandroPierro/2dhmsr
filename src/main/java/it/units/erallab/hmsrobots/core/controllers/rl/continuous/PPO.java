package it.units.erallab.hmsrobots.core.controllers.rl.continuous;

import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;
import static org.apache.commons.math3.linear.MatrixUtils.createRealVector;

public class PPO implements ContinuousRL {

  static class MLP {

    private final int inputDim;
    private final int outputDim;

    // Network parameters
    private RealMatrix W1;
    private RealMatrix W2;
    private RealMatrix W3;
    private RealVector b1;
    private RealVector b2;
    private RealVector b3;

    // Parameters gradient
    private RealMatrix dW1;
    private RealMatrix dW2;
    private RealMatrix dW3;
    private RealVector db1;
    private RealVector db2;
    private RealVector db3;

    // Network state
    private RealVector z1;
    private RealVector z2;

    MLP(int inputDim, int outputDim) {
      this.inputDim = inputDim;
      this.outputDim = outputDim;
      Random random = new Random();
      this.W1 = createRealMatrix(sampleMatrix(64, inputDim, random));
      this.W2 = createRealMatrix(sampleMatrix(64, 64, random));
      this.W3 = createRealMatrix(sampleMatrix(outputDim, 64, random));
      this.b1 = createRealVector(sampleVector(64, random));
      this.b2 = createRealVector(sampleVector(64, random));
      this.b3 = createRealVector(sampleVector(outputDim, random));
      this.dW1 = createRealMatrix(64, inputDim);
      this.dW2 = createRealMatrix(64, 64);
      this.dW3 = createRealMatrix(outputDim, 64);
      this.db1 = createRealVector(new double[64]);
      this.db2 = createRealVector(new double[64]);
      this.db3 = createRealVector(new double[outputDim]);
    }

    MLP(RealMatrix W1, RealMatrix W2, RealMatrix W3, RealVector b1, RealVector b2, RealVector b3) {
      this.inputDim = W1.getColumnDimension();
      this.outputDim = b3.getDimension();
      this.W1 = W1.copy();
      this.W2 = W2.copy();
      this.W3 = W3.copy();
      this.b1 = b1.copy();
      this.b2 = b2.copy();
      this.b3 = b3.copy();
    }

    public RealVector forward(RealVector x) {
      z1 = W1.operate(x).add(b1).map(Math::tanh);
      z2 = W2.operate(z1).add(b2).map(Math::tanh);
      return W3.operate(z2).add(b3);
    }

    public MLP copy() {
      return new MLP(W1, W2, W3, b1, b2, b3);
    }

    public void sgdStep(List<RealVector> x, List<RealVector> grad, double lr) {
      zeroGrad();
      for (int i = 0; i < x.size(); i++) {
        RealVector xi = x.get(i);
        RealVector gi = grad.get(i);
        this.forward(xi);
        db3 = db3.add(gi);
        dW3 = dW3.add(z2.outerProduct(gi).transpose());
        RealVector dz2 = W3.transpose().operate(gi);
        RealVector dl1 = dz2.ebeMultiply(z1.map(y -> (1 - y * y)));
        db2 = db2.add(dl1);
        dW2 = dW2.add(z1.outerProduct(dl1).transpose());
        RealVector dz1 = W2.transpose().operate(dl1);
        RealVector dl2 = dz1.ebeMultiply(z1.map(y -> (1 - y * y)));
        db1 = db1.add(dl2);
        dW1 = dW1.add(xi.outerProduct(dl2).transpose());
      }
      W1 = W1.add(dW1.scalarMultiply(lr));
      W2 = W2.add(dW2.scalarMultiply(lr));
      W3 = W3.add(dW3.scalarMultiply(lr));
      b1 = b1.add(db1.mapMultiply(lr));
      b2 = b2.add(db2.mapMultiply(lr));
      b3 = b3.add(db3.mapMultiply(lr));
    }

    public void zeroGrad() {
      this.dW1 = createRealMatrix(64, inputDim);
      this.dW2 = createRealMatrix(64, 64);
      this.dW3 = createRealMatrix(outputDim, 64);
      this.db1 = createRealVector(new double[64]);
      this.db2 = createRealVector(new double[64]);
      this.db3 = createRealVector(new double[outputDim]);
    }

    public void printWeights() {
      System.out.println("W1 : " + W1);
      System.out.println("W2 : " + W2);
      System.out.println("W3 : " + W3);
    }
  }

  private final int stateDim;
  private final int actionDim;

  private final Random random = new Random();

  // Networks
  private final MLP policy;
  private final MLP value;
  RealVector policyStd;

  // Internals
  private int t = 0;
  private final int maxT = 128;
  private final double learningRate = 0.05;
  private final int epochs = 5;
  private final int nBatch = 4;
  private final double[] rewards = new double[maxT];
  private final List<RealVector> states = new ArrayList<>();
  private final List<RealVector> actions = new ArrayList<>();

  public PPO(int stateDim, int actionDim) {
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.policy = new MLP(stateDim, actionDim);
    this.value = new MLP(stateDim, 1);
    this.policyStd = createRealVector(sampleVector(actionDim, random));
  }

  RealVector sampleAction(RealVector state) {
    RealVector means = policy.forward(state);
    return means.add(policyStd.map(x -> random.nextGaussian() * Math.exp(x)));
  }


  @Override
  public int getInputDimension() {
    return stateDim;
  }

  @Override
  public int getOutputDimension() {
    return actionDim;
  }

  @Override
  public void reset() {
    t = 0;
  }

  @Override
  public double[] apply(double time, double[] input, double r) {
    RealVector state = createRealVector(input);
    RealVector action = sampleAction(createRealVector(input));

    if (t == maxT) {
      for (int i = 0; i < epochs; i++) {
        // initialize solution as number 1 to maxT using streams
        List<Integer> solution = IntStream.range(0, maxT).boxed().collect(Collectors.toList());
        Collections.shuffle(solution);
        for (int j = 0; j < nBatch; j++) {
          List<RealVector> statesBatch = solution.stream().map(states::get).toList();
          List<RealVector> actionsBatch = solution.stream().map(actions::get).toList();
        }
      }
      t = 0;
      states.clear();
      actions.clear();
    }

    states.add(state);
    actions.add(action);

    rewards[t] = r;
    t += 1;
    return action.map(Math::tanh).toArray();
  }

  @Override
  public void reinitialize() {

  }


  private static double[][] sampleMatrix(int rows, int cols, Random random) {
    double[][] matrix = new double[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        matrix[i][j] = random.nextGaussian();
      }
    }
    return matrix.clone();
  }

  private static double[] sampleVector(int length, Random random) {
    double[] vector = new double[length];
    for (int i = 0; i < length; i++) {
      vector[i] = random.nextGaussian();
    }
    return vector.clone();
  }


  @Override
  public Snapshot getSnapshot() {
    return null;
  }
}
