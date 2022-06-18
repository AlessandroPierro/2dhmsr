package it.units.erallab.hmsrobots.core.controllers.dqn;

import it.units.erallab.hmsrobots.core.controllers.Resettable;

import java.io.Serializable;
import java.util.function.BiFunction;

public abstract class ContinuousRL implements Resettable, Serializable, BiFunction<double[], Double, double[]> {

}
