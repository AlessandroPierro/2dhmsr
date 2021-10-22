/*
 * Copyright (C) 2020 Eric Medvet <eric.medvet@gmail.com> (as Eric Medvet <eric.medvet@gmail.com>)
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package it.units.erallab.hmsrobots.util;

import com.google.common.collect.LinkedHashMultiset;
import com.google.common.collect.Multiset;
import org.apache.commons.lang3.tuple.Pair;

import java.util.*;
import java.util.function.Predicate;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * @author Eric Medvet <eric.medvet@gmail.com>
 */
public class Utils {

  private Utils() {
  }

  private static final Logger L = Logger.getLogger(Utils.class.getName());

  public static <K> Grid<K> gridLargestConnected(Grid<K> kGrid, Predicate<K> p) {
    Grid<Integer> iGrid = partitionGrid(kGrid, p);
    //count elements per partition
    Multiset<Integer> counts = LinkedHashMultiset.create();
    for (Integer i : iGrid.values()) {
      if (i != null) {
        counts.add(i);
      }
    }
    //find largest
    Integer maxIndex = null;
    int max = 0;
    for (Integer index : counts.elementSet()) {
      int count = counts.count(index);
      if (count > max) {
        max = count;
        maxIndex = index;
      }
    }
    //if no largest, return empty
    if (maxIndex == null) {
      return Grid.create(kGrid);
    }
    //filter map
    Grid<K> filtered = Grid.create(kGrid);
    for (Grid.Entry<Integer> iEntry : iGrid) {
      if (iEntry.getValue() != null && iEntry.getValue().equals(maxIndex)) {
        filtered.set(iEntry.getX(), iEntry.getY(), kGrid.get(iEntry.getX(), iEntry.getY()));
      }
    }
    return filtered;
  }

  public static <K> Grid<K> gridConnected(Grid<K> kGrid, Comparator<K> comparator, int n) {
    Comparator<Grid.Entry<K>> entryComparator = (e1, e2) -> comparator.compare(e1.getValue(), e2.getValue());
    Predicate<Pair<Grid.Entry<?>, Grid.Entry<?>>> adjacencyPredicate = p -> (Math.abs(p.getLeft().getX() - p.getRight().getX()) <= 1 && p.getLeft().getY() == p.getRight().getY()) || (Math.abs(p.getLeft().getY() - p.getRight().getY()) <= 1 && p.getLeft().getX() == p.getRight().getX());
    Grid.Entry<K> entryFirst = kGrid.stream()
        .min(entryComparator)
        .orElseThrow(() -> new IllegalArgumentException("Grid has no max element"));
    Set<Grid.Entry<K>> selected = new HashSet<>(n);
    selected.add(entryFirst);
    while (selected.size() < n) {
      Set<Grid.Entry<K>> candidates = kGrid.stream()
          .filter(e -> e.getValue() != null)
          .filter(e -> !selected.contains(e))
          .filter(e -> selected.stream().anyMatch(f -> adjacencyPredicate.test(Pair.of(e, f))))
          .collect(Collectors.toSet());
      if (candidates.isEmpty()) {
        break;
      }
      selected.add(candidates.stream().min(entryComparator).orElse(entryFirst));
    }
    Grid<K> outGrid = Grid.create(kGrid.getW(), kGrid.getH());
    selected.forEach(e -> outGrid.set(e.getX(), e.getY(), e.getValue()));
    return outGrid;
  }
  
  private static <K> Grid<Integer> partitionGrid(Grid<K> kGrid, Predicate<K> p) {
    Grid<Integer> iGrid = Grid.create(kGrid);
    for (int x = 0; x < kGrid.getW(); x++) {
      for (int y = 0; y < kGrid.getH(); y++) {
        if (iGrid.get(x, y) == null) {
          int index = iGrid.values().stream().filter(Objects::nonNull).mapToInt(i -> i).max().orElse(0);
          partitionGrid(x, y, index + 1, kGrid, iGrid, p);
        }
      }
    }
    return iGrid;
  }

  private static <K> void partitionGrid(int x, int y, int i, Grid<K> kGrid, Grid<Integer> iGrid, Predicate<K> p) {
    boolean hereFilled = p.test(kGrid.get(x, y));
    //already done
    if (iGrid.get(x, y) != null) {
      return;
    }
    //filled but not done
    if (hereFilled) {
      iGrid.set(x, y, i);
      //expand east
      if (x > 0) {
        partitionGrid(x - 1, y, i, kGrid, iGrid, p);
      }
      //expand west
      if (x < kGrid.getW() - 1) {
        partitionGrid(x + 1, y, i, kGrid, iGrid, p);
      }
      //expand north
      if (y > 0) {
        partitionGrid(x, y - 1, i, kGrid, iGrid, p);
      }
      //expand south
      if (y < kGrid.getH() - 1) {
        partitionGrid(x, y + 1, i, kGrid, iGrid, p);
      }
    }
  }

  public static <K> Grid<K> cropGrid(Grid<K> inGrid, Predicate<K> p) {
    //find bounds
    int minX = inGrid.getW();
    int maxX = 0;
    int minY = inGrid.getH();
    int maxY = 0;
    for (Grid.Entry<K> entry : inGrid) {
      if (p.test(entry.getValue())) {
        minX = Math.min(minX, entry.getX());
        maxX = Math.max(maxX, entry.getX());
        minY = Math.min(minY, entry.getY());
        maxY = Math.max(maxY, entry.getY());
      }
    }
    //build new grid
    Grid<K> outGrid = Grid.create(maxX - minX + 1, maxY - minY + 1);
    //fill
    for (int x = 0; x < outGrid.getW(); x++) {
      for (int y = 0; y < outGrid.getH(); y++) {
        outGrid.set(x, y, inGrid.get(x + minX, y + minY));
      }
    }
    return outGrid;
  }

  public static double shapeElongation(Grid<Boolean> posture) {
    if (posture.values().stream().noneMatch(e -> e)) {
      throw new IllegalArgumentException("Grid is empty");
    }
    // find largest x-thickness and y-thickness by compacting everything on one axis and on the other
    int minX = Integer.MAX_VALUE;
    int maxX = Integer.MIN_VALUE;
    int minY = Integer.MAX_VALUE;
    int maxY = Integer.MIN_VALUE;
    for (Grid.Entry<Boolean> entry : posture) {
      if (entry.getValue()) {
        int y = (int) posture.stream().filter(e -> e.getValue() && e.getX() == entry.getX() && e.getY() < entry.getY()).count();//entry.getX();
        int x = (int) posture.stream().filter(e -> e.getValue() && e.getY() == entry.getY() && e.getX() < entry.getX()).count();//entry.getX();
        minX = Math.min(x, minX);
        maxX = Math.max(x, maxX);
        minY = Math.min(y, minY);
        maxY = Math.max(y, maxY);
      }
    }
    // compute ratio of the smallest thickness to the largest one
    int sideX = maxX - minX + 1;
    int sideY = maxY - minY + 1;
    double ratio = (double) Math.min(sideX, sideY) / Math.max(sideX, sideY);
    // take 1 - ratio to have 0.0 for smallest elongation (square or round shape) and 1.0 for largest (stick)
    return 1.0 - ratio;
  }

  @SafeVarargs
  public static <E> List<E> ofNonNull(E... es) {
    List<E> list = new ArrayList<>();
    for (E e : es) {
      if (e != null) {
        list.add(e);
      }
    }
    return list;
  }

  public static String param(String pattern, String string, String paramName) {
    Matcher matcher = Pattern.compile(pattern).matcher(string);
    if (matcher.matches()) {
      return matcher.group(paramName);
    }
    throw new IllegalStateException(String.format("Param %s not found in %s with pattern %s", paramName, string, pattern));
  }

  public static Map<String, String> params(String pattern, String string) {
    if (!string.matches(pattern)) {
      return null;
    }
    Matcher m = Pattern.compile("\\(\\?<([a-zA-Z][a-zA-Z0-9]*)>").matcher(pattern);
    List<String> groupNames = new ArrayList<>();
    while (m.find()) {
      groupNames.add(m.group(1));
    }
    Map<String, String> params = new HashMap<>();
    for (String groupName : groupNames) {
      String value = param(pattern, string, groupName);
      if (value != null) {
        params.put(groupName, value);
      }
    }
    return params;
  }

}
