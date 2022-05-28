package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import it.units.erallab.hmsrobots.core.controllers.IOSized;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;

import java.io.Serializable;
import java.util.function.Function;

public interface DiscreteRL extends IOSized, Snapshottable, Resettable, Serializable {
  interface InputConverter extends Function<double[], Integer>, IOSized {
  }

  interface OutputConverter extends Function<Integer, double[]>, IOSized {
  }

  int apply(double t, int input, double r);

  void reinitialize();

  default ContinuousRL with(InputConverter inputConverter, OutputConverter outputConverter) {
    DiscreteRL inner = this;
    return new ContinuousRL() {
      @Override
      public double[] apply(double t, double[] state, double reward) {
        return outputConverter.apply(inner.apply(t, inputConverter.apply(state), reward));
      }

      @Override
      public int getInputDimension() {
        return inputConverter.getInputDimension();
      }

      @Override
      public int getOutputDimension() {
        return outputConverter.getOutputDimension();
      }

      @Override
      public Snapshot getSnapshot() {
        return inner.getSnapshot();
      }

      @Override
      public void reset() {
        inner.reset();
      }

      @Override
      public void reinitialize() {
        inner.reinitialize();
      }
    };
  }

}
