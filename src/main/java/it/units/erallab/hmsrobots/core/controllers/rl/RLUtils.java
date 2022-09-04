package it.units.erallab.hmsrobots.core.controllers.rl;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class RLUtils {

  public static class InputConverter implements Function<double[], Integer>, Serializable {

    @JsonProperty
    private final int inputDimension;
    @JsonProperty
    private final int nPartitions;

    public InputConverter(
        @JsonProperty("ouputDimension") int inputDimension, @JsonProperty("nPartitions") int nPartitions
    ) {
      this.inputDimension = inputDimension;
      this.nPartitions = nPartitions;
    }

    public InputConverter(int inputDimension) {
      this(inputDimension, 2);
    }

    @Override
    public Integer apply(double[] vector) {
      int value = 0;
      for (int i = 0; i < inputDimension; i++) {
        //if (vector[i] < 0d || vector[i] > 1d) {
        //  throw new IllegalArgumentException("RLUtils.InputConverter: input vector must be in [0, 1], received " + Arrays.toString(
        //      vector));
        //}
        int idx = Math.min((int) (vector[i] * nPartitions), nPartitions - 1);
        value += idx * Math.pow(nPartitions, i);
      }
      return value;
    }

    public int getMaxOutput() {
      return (int) Math.pow(nPartitions, inputDimension) - 1;
    }
  }

  public static class OutputConverter implements Function<Integer, double[]>, Serializable {

    @JsonProperty
    private final int outputDimension;
    @JsonProperty
    private final int nPartitions;
    @JsonProperty
    private final double maxValue;

    public OutputConverter(
        @JsonProperty("outputDimension") int outputDimension,
        @JsonProperty("nPartitions") int nPartitions,
        @JsonProperty("maxValue") double maxValue
    ) {
      this.outputDimension = outputDimension;
      this.nPartitions = nPartitions;
      this.maxValue = maxValue;
    }

    public OutputConverter(int outputDimension) {
      this(outputDimension, 2, 1d);
    }

    @Override
    public double[] apply(Integer action) {
      //if (action > getMaxInput() || action < 0) {
      //  throw new IllegalArgumentException("RL OutputConverter - Invalid input value: " + action);
      //}
      double[] output = new double[outputDimension];
      double delta = 2 * maxValue / (nPartitions - 1);
      String actionString = Integer.toString(action, nPartitions);
      actionString = String.format("%1$" + outputDimension + "s", actionString).replace(' ', '0');
      for (int i = 0; i < outputDimension; i++) {
        output[i] = -maxValue + delta * Integer.parseInt(String.valueOf(actionString.charAt(i)));
      }
      return output;
    }

    public int getMaxInput() {
      return (int) Math.pow(nPartitions, outputDimension) - 1;
    }

  }

  public record RLTransition(double[] state, int action, double reward, double[] nextState) {}

  public static class ReplayMemory {
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
      // TODO : implement weighted sampling based on error
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

}
