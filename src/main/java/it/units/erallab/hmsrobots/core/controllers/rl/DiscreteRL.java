package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.IOSized;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;

import java.util.function.Function;

public interface DiscreteRL extends IOSized, Snapshottable, Resettable {
  interface InputConverter extends Function<double[], Integer>, IOSized {
  }

  interface OutputConverter extends Function<Integer, double[]>, IOSized {
  }

  int apply(double t, int input, double r);
  void stopExploration();

  default ContinuousRL with(InputConverter inputConverter, OutputConverter outputConverter) {
    DiscreteRL inner = this;
    return new ContinuousRL() {
      @Override
      public void reset() { }

      @Override
      public Snapshot getSnapshot() {
        return inner.getSnapshot();
      }

      @Override
      public double[] apply(double t, double[] state, double reward) {
        return outputConverter.apply(inner.apply(t, inputConverter.apply(state), reward));
      }

      @Override
      public void stopExploration() { }

      @Override
      public int getInputDimension() {
        return inputConverter.getInputDimension();
      }

      @Override
      public int getOutputDimension() {
        return outputConverter.getOutputDimension();
      }
    };
  }

}
