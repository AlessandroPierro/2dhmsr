package it.units.erallab.hmsrobots.core.controllers.rl;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.*;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializableBiFunction;

import java.io.Serializable;
import java.util.*;
import java.util.function.ToDoubleFunction;

public class ClusteredObservationFunction implements SerializableBiFunction<Double, Grid<Voxel>, double[]> {

    public record Config (
            boolean area,
            boolean touch,
            boolean rotation,
            boolean velocityX,
            boolean velocityY,
            boolean accelerationX,
            boolean accelerationY
    ) implements Serializable {}

    @JsonProperty
    private final List<List<Grid.Key>> clusters;
    @JsonProperty
    private final Config config;
    @JsonProperty
    private final int nSteps;

    private transient List<double[]> history;

    private transient LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map;

    private transient int nSensorReadings;

    @JsonCreator
    public ClusteredObservationFunction(
            @JsonProperty("clusters") List<List<Grid.Key>> clusters,
            @JsonProperty("config") Config config,
            @JsonProperty("nSteps") int nSteps
    ) {
        this.clusters = clusters;
        this.config = config;
        this.map = makeMap();
        this.nSteps = nSteps;
        this.nSensorReadings = nSteps * map.values().stream().mapToInt(Map::size).sum();
        this.history = new ArrayList<>();
    }

    @JsonCreator
    public ClusteredObservationFunction(
            @JsonProperty("clusters") List<List<Grid.Key>> clusters,
            @JsonProperty("config") Config config
    ) {
        this(clusters, config, 1);
    }

    @Override
    public double[] apply(
            Double t, Grid<Voxel> voxels
    ) {

        if (map == null) {
            this.map = makeMap();
            this.nSensorReadings = map.values().stream().mapToInt(Map::size).sum();
        }

        if (history == null) {
            this.history = new ArrayList<>();
        }

        double[] observations = new double[nSensorReadings];

        int counter = 0;

        for (Map.Entry<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> entry : map.entrySet()) {

            List<Grid.Key> cluster = entry.getKey();
            Map<Class<? extends Sensor>, ToDoubleFunction<double[]>> sensorMap = entry.getValue();

            for (Map.Entry<Class<? extends Sensor>, ToDoubleFunction<double[]>> sensorEntry : sensorMap.entrySet()) {

                Class<? extends Sensor> sensorType = sensorEntry.getKey();
                ToDoubleFunction<double[]> aggregationFunction = sensorEntry.getValue();

                double[] temp = new double[cluster.size()];

                for (int i = 0; i < cluster.size(); i++) {
                    Grid.Key key = cluster.get(i);
                    List<Sensor> sensors = voxels.get(key.x(), key.y())
                            .getSensors()
                            .stream()
                            .filter(s -> (s instanceof CompositeSensor cs ? cs.getInnermostSensor() : s).getClass()
                                    .isAssignableFrom(sensorType)).toList();
                    temp[i] = aggregationFunction
                            .applyAsDouble(sensors.stream().mapToDouble(s -> s.getReadings()[0]).toArray());
                }

                observations[counter] = aggregationFunction.applyAsDouble(temp);
                counter++;

            }

        }

        history.add(observations.clone());
        if (history.size() > nSteps) {
            history.remove(0);
        }

        // convert List<double[]> to double[]
        if (nSteps == 1) {
            return observations;
        }
        return history.stream().flatMapToDouble(Arrays::stream).toArray();
    }

    private LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> makeMap() {
        LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map = new LinkedHashMap<>();
        for (List<Grid.Key> cluster : clusters) {
            LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>> sensorMap = new LinkedHashMap<>();
            ToDoubleFunction<double[]> meanOp = x -> x.length == 0 ? 0d : Arrays.stream(x).sum() / x.length;
            ToDoubleFunction<double[]> max = x -> Arrays.stream(x).max().orElse(0d);
            ToDoubleFunction<double[]> min = x -> Arrays.stream(x).min().orElse(0d);
            ToDoubleFunction<double[]> angleMap = x -> 0.25 < min.applyAsDouble(x) && max.applyAsDouble(x) < 0.75 ? 0.45 : 0.85;
            if (config.area)
                sensorMap.put(AreaRatio.class, meanOp);
            if (config.touch)
                sensorMap.put(Touch.class, max);
            if (config.rotation)
                sensorMap.put(Angle.class, angleMap);
            if (config.velocityX)
                sensorMap.put(VelocityX.class, meanOp);
            if (config.velocityY)
                sensorMap.put(VelocityY.class, meanOp);
            if (config.accelerationX)
                sensorMap.put(AccelerationX.class, meanOp);
            if (config.accelerationY)
                sensorMap.put(AccelerationY.class, meanOp);
            map.put(cluster, sensorMap);
        }
        return map;
    }

    public int getOutputDimension() {
        return nSensorReadings;
    }
}