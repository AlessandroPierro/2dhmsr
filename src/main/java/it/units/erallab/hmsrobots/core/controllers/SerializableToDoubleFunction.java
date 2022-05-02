package it.units.erallab.hmsrobots.core.controllers;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.io.Serializable;
import java.util.function.ToDoubleFunction;

public interface SerializableToDoubleFunction<T> extends ToDoubleFunction<T>, Serializable {
}
