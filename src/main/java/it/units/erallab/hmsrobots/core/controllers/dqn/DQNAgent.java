package it.units.erallab.hmsrobots.core.controllers.dqn;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

public class DQNAgent extends ContinuousRL {

  public DQNAgent(int stateDimension, int actionDimension) {
    this.stateDimension = stateDimension;
    this.actionDimension = actionDimension;
    this.network = new DQN(stateDimension, (int) (0.7 * stateDimension), (int) (0.7 * stateDimension), actionDimension);
    this.targetNetwork = network.copy();
    this.replayMemory = new ReplayMemory(REPLAY_MEMORY_SIZE);
    this.random = new Random(RANDOM_SEED);
    this.outputConverter = new OutputConverter(actionDimension);
  }

  public DQNAgent(
      @JsonProperty("stateDimension") int stateDimension,
      @JsonProperty("actionDimension") int actionDimension,
      @JsonProperty("network") DQN network,
      @JsonProperty("outputConverter") OutputConverter outputConverter
  ) {
    this.stateDimension = stateDimension;
    this.actionDimension = actionDimension;
    this.network = network;
    this.outputConverter = outputConverter;
  }

  private record RLTransition(double[] state, int action, double reward, double[] nextState) {}

  private static class ReplayMemory {
    private final int capacity;
    private final List<RLTransition> memory;
    private final Random random;
    private static final int RANDOM_SEED = 42;

    public ReplayMemory(int capacity) {
      this.capacity = capacity;
      this.memory = new ArrayList<>();
      this.random = new Random(RANDOM_SEED);
    }

    public void add(RLTransition transition) {
      memory.add(transition);
      if (memory.size() > capacity) {
        memory.remove(0);
      }
    }

    public List<RLTransition> sampleBatch(int batchSize) {
      if (batchSize > memory.size()) {
        throw new IllegalArgumentException("ReplayMemory - Batch size cannot be greater than the memory size");
      }
      for (int i = memory.size() - 1; i >= memory.size() - batchSize; --i) {
        Collections.swap(memory, i, random.nextInt(i + 1));
      }
      return memory.subList(memory.size() - batchSize, memory.size());
    }

    public void empty() {
      memory.clear();
    }

    public int getSize() {
      return memory.size();
    }

  }

  private static class DQN implements Function<double[], double[]>, Serializable {
    @JsonProperty
    private final int inputDimension;
    @JsonProperty
    private final int layer1Dimension;
    @JsonProperty
    private final int layer2Dimension;
    @JsonProperty
    private final int outputDimension;
    @JsonProperty
    private final double[][] W1;
    @JsonProperty
    private final double[][] W2;
    @JsonProperty
    private final double[][] W3;
    @JsonProperty
    private final double[] b1;
    @JsonProperty
    private final double[] b2;
    @JsonProperty
    private final double[] b3;

    private static final double LEARNING_RATE = 0.001;
    private static final int RANDOM_SEED = 42;

    private transient double[] inputState;
    private transient double[] layer1Output;
    private transient double[] layer2Output;

    public DQN(int inputDimension, int layer1Dimension, int layer2Dimension, int outputDimension) {
      this.inputDimension = inputDimension;
      this.layer1Dimension = layer1Dimension;
      this.layer2Dimension = layer2Dimension;
      this.outputDimension = outputDimension;
      this.W1 = new double[layer1Dimension][inputDimension];
      this.W2 = new double[layer2Dimension][layer1Dimension];
      this.W3 = new double[outputDimension][layer2Dimension];
      this.b1 = new double[layer1Dimension];
      this.b2 = new double[layer2Dimension];
      this.b3 = new double[outputDimension];
      initializeWeights();
    }

    public DQN(
        @JsonProperty("inputDimension") int inputDimension,
        @JsonProperty("layer1Dimension") int layer1Dimension,
        @JsonProperty("layer2Dimension") int layer2Dimension,
        @JsonProperty("outputDimension") int outputDimension,
        @JsonProperty("W1") double[][] W1,
        @JsonProperty("W2") double[][] W2,
        @JsonProperty("W3") double[][] W3,
        @JsonProperty("b1") double[] b1,
        @JsonProperty("b2") double[] b2,
        @JsonProperty("b3") double[] b3
    ) {
      this.inputDimension = inputDimension;
      this.layer1Dimension = layer1Dimension;
      this.layer2Dimension = layer2Dimension;
      this.outputDimension = outputDimension;
      this.W1 = W1;
      this.W2 = W2;
      this.W3 = W3;
      this.b1 = b1;
      this.b2 = b2;
      this.b3 = b3;
    }

    @Override
    public double[] apply(double[] input) {
      double[] output = new double[outputDimension];
      this.inputState = input;
      this.layer1Output = new double[layer1Dimension];
      for (int i = 0; i < layer1Dimension; ++i) {
        double sum = 0.0;
        for (int j = 0; j < inputDimension; ++j) {
          sum += W1[i][j] * input[j];
        }
        layer1Output[i] = Math.max(0d, sum + b1[i]);
      }
      this.layer2Output = new double[layer2Dimension];
      for (int i = 0; i < layer2Dimension; ++i) {
        double sum = 0.0;
        for (int j = 0; j < layer1Dimension; ++j) {
          sum += W2[i][j] * layer1Output[j];
        }
        layer2Output[i] = Math.max(0d, sum + b2[i]);
      }
      for (int i = 0; i < outputDimension; ++i) {
        double sum = 0.0;
        for (int j = 0; j < layer2Dimension; ++j) {
          sum += W3[i][j] * layer2Output[j];
        }
        output[i] = Math.max(0d, sum + b3[i]);
      }
      return output;
    }

    void initializeWeights() {
      Random random = new Random(RANDOM_SEED);
      double normalization = 1.0 / Math.sqrt(inputDimension);
      for (int i = 0; i < layer1Dimension; ++i) {
        for (int j = 0; j < inputDimension; ++j) {
          W1[i][j] = random.nextDouble() * 2d * normalization - normalization;
        }
        b1[i] = random.nextDouble() * 2d * normalization - normalization;
      }
      normalization = 1.0 / Math.sqrt(layer1Dimension);
      for (int i = 0; i < layer2Dimension; ++i) {
        for (int j = 0; j < layer1Dimension; ++j) {
          W2[i][j] = random.nextDouble() * 2d * normalization - normalization;
        }
        b2[i] = random.nextDouble() * 2d * normalization - normalization;
      }
      normalization = 1.0 / Math.sqrt(layer2Dimension);
      for (int i = 0; i < outputDimension; ++i) {
        for (int j = 0; j < layer2Dimension; ++j) {
          W3[i][j] = random.nextDouble() * 2d * normalization - normalization;
        }
        b3[i] = random.nextDouble() * 2d * normalization - normalization;
      }
    }

    public int getInputDimension() {
      return inputDimension;
    }

    public int getOutputDimension() {
      return outputDimension;
    }

    public int getParameterCount() {
      return W1.length * W1[0].length + W2.length * W2[0].length + W3.length * W3[0].length + b1.length + b2.length + b3.length;
    }

    public void updateWeights(List<RLTransition> batch, DQN targetDQN) {
      double[][] dW1 = new double[layer1Dimension][inputDimension];
      double[][] dW2 = new double[layer2Dimension][layer1Dimension];
      double[][] dW3 = new double[outputDimension][layer2Dimension];
      double[] db1 = new double[layer1Dimension];
      double[] db2 = new double[layer2Dimension];
      double[] db3 = new double[outputDimension];
      for (RLTransition transition : batch) {
        double[] target = targetDQN.apply(transition.state());
        double targetValue = target[transition.action()];
        double[] output = apply(transition.state());
        double value = output[transition.action()];
        double error = targetValue - value;
        db3[transition.action()] += Math.max(0d, 2d * error);
        for (int i = 0; i < layer2Dimension; ++i) {
          dW3[transition.action()][i] += Math.max(0d, 2d * error) * layer2Output[i];
        }
        for (int i = 0; i < layer2Dimension; ++i) {
          db2[i] += Math.max(0d, 2d * error) * W3[transition.action()][i];
          for (int j = 0; j < layer1Dimension; ++j) {
            dW2[i][j] += Math.max(0d, 2d * error) * W3[transition.action()][i] * layer1Output[j];
          }
        }
        for (int i = 0; i < layer1Dimension; ++i) {
          db1[i] += Math.max(0d, 2d * error) * W2[transition.action()][i];
          for (int j = 0; j < inputDimension; ++j) {
            dW1[i][j] += Math.max(0d, 2d * error) * W2[transition.action()][i] * inputState[j];
          }
        }
      }

      for (int i = 0; i < layer1Dimension; ++i) {
        for (int j = 0; j < inputDimension; ++j) {
          W1[i][j] += LEARNING_RATE * dW1[i][j];
        }
        b1[i] += LEARNING_RATE * db1[i];
      }
      for (int i = 0; i < layer2Dimension; ++i) {
        for (int j = 0; j < layer1Dimension; ++j) {
          W2[i][j] += LEARNING_RATE * dW2[i][j];
        }
        b2[i] += LEARNING_RATE * db2[i];
      }
      for (int i = 0; i < outputDimension; ++i) {
        for (int j = 0; j < layer2Dimension; ++j) {
          W3[i][j] += LEARNING_RATE * dW3[i][j];
        }
        b3[i] += LEARNING_RATE * db3[i];
      }
    }

    public DQN copy() {
      return new DQN(
          inputDimension,
          layer1Dimension,
          layer2Dimension,
          outputDimension,
          copy(W1),
          copy(W2),
          copy(W3),
          b1.clone(),
          b2.clone(),
          b3.clone()
      );
    }

    private static double[][] copy(double[][] array) {
      double[][] copy = new double[array.length][];
      for (int i = 0; i < array.length; ++i) {
        copy[i] = array[i].clone();
      }
      return copy;
    }
  }

  private static class OutputConverter implements Function<Integer, double[]>, Serializable {
    @JsonProperty
    private final int outputDimension;

    public OutputConverter(@JsonProperty("ouputDimension") int outputDimension) {
      this.outputDimension = outputDimension;
    }

    @Override
    public double[] apply(Integer action) {
      double[] output = new double[outputDimension];
      for (int i = 0; i < outputDimension; i++) {
        output[i] = action % (int) Math.pow(3, i + 1) == 0 ? -0.75 : action % (int) Math.pow(3, i + 1) == 1 ? 0.75 : 0.0;
      }
      return output;
    }
  }

  static final int REPLAY_MEMORY_SIZE = 50000;
  static final int BATCH_SIZE = 32;
  static final int POLICY_UPDATE_FREQUENCY = 4;
  static final int RANDOM_SEED = 42;

  @JsonProperty
  protected final int stateDimension;
  @JsonProperty
  protected final int actionDimension;
  @JsonProperty
  protected final DQN network;
  @JsonProperty
  protected final OutputConverter outputConverter;

  // Internal variables
  protected transient Random random;
  protected transient DQN targetNetwork;
  protected transient ReplayMemory replayMemory;
  protected transient double[] previousState;
  protected transient int previousAction;
  protected transient boolean enableLearning = true;
  protected transient int stepPolicyUpdate = 0;
  protected transient int stepAction = 1;

  @Override
  public double[] apply(double[] state, Double reward) {
    if (targetNetwork == null) {
      random = new Random(RANDOM_SEED);
      targetNetwork = network.copy();
      replayMemory = new ReplayMemory(REPLAY_MEMORY_SIZE);
      previousState = null;
      previousAction = -1;
      enableLearning = true;
      stepPolicyUpdate = 0;
      stepAction = 1;
    }
    if (previousState != null) {
      replayMemory.add(new RLTransition(previousState.clone(), previousAction, reward, state.clone()));
    }
    if (enableLearning && replayMemory.getSize() > BATCH_SIZE) {
      updateNetwork();
      stepPolicyUpdate++;
      if (stepPolicyUpdate >= POLICY_UPDATE_FREQUENCY) {
        stepPolicyUpdate = 0;
        targetNetwork = network.copy();
      }
    }
    previousState = state.clone();
    previousAction = selectAction(state);
    stepAction++;
    return outputConverter.apply(previousAction);
  }

  private int selectAction(double[] state) {
    int action;
    if (random.nextDouble() < getExplorationRate()) {
      action = random.nextInt(actionDimension);
    } else {
      double[] output = network.apply(state);
      action = 0;
      for (int i = 1; i < actionDimension; ++i) {
        if (output[i] > output[action]) {
          action = i;
        }
      }
    }
    return action;
  }

  @Override
  public void reset() {
    previousState = null;
    previousAction = -1;
  }

  private void updateNetwork() {
    List<RLTransition> batch = replayMemory.sampleBatch(BATCH_SIZE);
    network.updateWeights(batch, targetNetwork);
  }

  private double getExplorationRate() {
    return Math.max(0.1, 1d - 0.9 * 0.0000001 * stepAction);
  }

  public void enableLearning() {
    enableLearning = true;
  }

  public void disableLearning() {
    enableLearning = false;
  }
}
