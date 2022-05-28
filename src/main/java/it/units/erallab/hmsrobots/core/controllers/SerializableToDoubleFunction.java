package it.units.erallab.hmsrobots.core.controllers;

import java.io.Serializable;
import java.util.function.ToDoubleFunction;

public interface SerializableToDoubleFunction<T> extends ToDoubleFunction<T>, Serializable {
}
