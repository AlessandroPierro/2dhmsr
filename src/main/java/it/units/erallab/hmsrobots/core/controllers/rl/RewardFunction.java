package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializableFunction;

public interface RewardFunction extends Resettable, SerializableFunction<Grid<Voxel>, Double> {

  @Override
  default void reset() {
  }
}
