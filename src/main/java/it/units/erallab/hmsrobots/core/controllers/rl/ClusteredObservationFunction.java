package it.units.erallab.hmsrobots.core.controllers.rl;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.*;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializableBiFunction;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;

public class ClusteredObservationFunction implements SerializableBiFunction<Double, Grid<Voxel>, double[]> {

    @JsonProperty
    private final List<List<Grid.Key>> clusters;
    @JsonProperty
    private final boolean area;
    @JsonProperty
    private final boolean touch;
    @JsonProperty
    private final boolean rotation;

    private transient LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map;

    private final int nSensorReadings;

    @JsonCreator
    public ClusteredObservationFunction(
            @JsonProperty("clusters") List<List<Grid.Key>> clusters,
            @JsonProperty("area") boolean area,
            @JsonProperty("touch") boolean touch,
            @JsonProperty("rotation") boolean rotation
    ) {
        this.clusters = clusters;
        this.area = area;
        this.touch = touch;
        this.rotation = rotation;
        this.map = makeMap();
        this.nSensorReadings = map.values().stream().mapToInt(Map::size).sum();
    }

    @Override
    public double[] apply(
            Double t, Grid<Voxel> voxels
    ) {

        if (map == null) {
            map = makeMap();
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
        return observations;
    }

    private LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> makeMap() {
        LinkedHashMap<List<Grid.Key>, LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>>> map = new LinkedHashMap<>();
        int counter = 0;
        for (List<Grid.Key> cluster : clusters) {
            LinkedHashMap<Class<? extends Sensor>, ToDoubleFunction<double[]>> sensorMap = new LinkedHashMap<>();
            ToDoubleFunction<double[]> meanOp = x -> x.length == 0 ? 0d : Arrays.stream(x).sum() / x.length;
            ToDoubleFunction<double[]> max = x -> Arrays.stream(x).max().orElse(0d);
            ToDoubleFunction<double[]> min = x -> Arrays.stream(x).min().orElse(0d);
            ToDoubleFunction<double[]> angleMap = x -> 0.25 < min.applyAsDouble(x) && max.applyAsDouble(x) < 0.75 ? 0.45 : 0.85;
            if (area)
                sensorMap.put(AreaRatio.class, meanOp);
            if (touch)
                sensorMap.put(Touch.class, max);
            if (rotation && counter == 0) {
                sensorMap.put(Angle.class, angleMap);
                counter++;
            }
            map.put(cluster, sensorMap);
        }
        return map;
    }

    public int getOutputDimension() {
        return nSensorReadings;
    }
}