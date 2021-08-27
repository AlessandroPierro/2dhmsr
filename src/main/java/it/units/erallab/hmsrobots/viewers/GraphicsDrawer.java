/*
 * Copyright (C) 2021 Eric Medvet <eric.medvet@gmail.com> (as Eric Medvet <eric.medvet@gmail.com>)
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
package it.units.erallab.hmsrobots.viewers;

import it.units.erallab.hmsrobots.core.geometry.BoundingBox;
import it.units.erallab.hmsrobots.core.geometry.Point2;
import it.units.erallab.hmsrobots.core.geometry.Poly;
import it.units.erallab.hmsrobots.core.objects.Ground;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.util.Configurable;
import it.units.erallab.hmsrobots.util.ConfigurableField;
import it.units.erallab.hmsrobots.viewers.drawers.*;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Eric Medvet <eric.medvet@gmail.com>
 */
public class GraphicsDrawer implements Configurable<GraphicsDrawer> {

  private final static PolyDrawer MINIATURE_GROUND_DRAWER = PolyDrawer.build();
  private final static BoundingBox MINIATURE_REL_BOUNDING_BOX = BoundingBox.build(Point2.build(0.65, 0.01), Point2.build(0.99, 0.35));

  static {
    MINIATURE_GROUND_DRAWER.setConfigurable("useTexture", false);
    MINIATURE_GROUND_DRAWER.setConfigurable("strokeColor", null);
  }

  public static final List<Drawer> LOW_DETAIL_DRAWERS = List.of(
      new PolyDrawer(Ground.class),
      new VoxelDrawer()
  );

  public static final List<Drawer> MEDIUM_DETAIL_DRAWERS = List.of(
      new PolyDrawer(Ground.class),
      new VoxelDrawer(),
      new SensorReadingsSectorDrawer(),
      new LidarDrawer()
  );

  public enum GeneralRenderingMode {
    GRID_MAJOR, GRID_MINOR, VIEWPORT_INFO, TIME_INFO, ROBOT_CENTERS_INFO
  }

  @ConfigurableField(uiType = ConfigurableField.Type.BASIC, enumClass = GeneralRenderingMode.class)
  private Set<GeneralRenderingMode> generalRenderingModes = new HashSet<>(Set.of(
      GeneralRenderingMode.ROBOT_CENTERS_INFO,
      GeneralRenderingMode.TIME_INFO
  ));
  @ConfigurableField
  private Color gridColor = Color.GRAY;
  @ConfigurableField
  private Color infoColor = Color.BLUE;
  @ConfigurableField
  private Color backgroundColor = Color.WHITE;
  @ConfigurableField
  private Color basicColor = Color.BLUE;
  private double[] gridSizes = new double[]{2, 5, 10};
  @ConfigurableField(uiMin = 1, uiMax = 5)
  private float strokeWidth = 1f;
  @ConfigurableField
  private boolean drawMiniature = true;
  @ConfigurableField(uiMin = 1, uiMax = 10)
  private float miniatureMagnifyRatio = 3f;
  @ConfigurableField(uiType = ConfigurableField.Type.BASIC)
  private List<Drawer> drawers = new ArrayList<>(MEDIUM_DETAIL_DRAWERS);

  private GraphicsDrawer() {
  }

  public static GraphicsDrawer build() {
    return new GraphicsDrawer();
  }

  /*public void drawMiniature(SnapshotOLD snapshot, Graphics2D g, it.units.erallab.hmsrobots.core.geometry.BoundingBox graphicsFrame, it.units.erallab.hmsrobots.core.geometry.BoundingBox inWorldFrame) {
    //set clipping area
    g.setClip(
        (int) graphicsFrame.min.x, (int) graphicsFrame.min.y,
        (int) graphicsFrame.width(), (int) graphicsFrame.height()
    );
    it.units.erallab.hmsrobots.core.objects.immutable.Ground ground = (it.units.erallab.hmsrobots.core.objects.immutable.Ground) snapshot.getObjects().stream().filter(i -> i instanceof it.units.erallab.hmsrobots.core.objects.immutable.Ground).findFirst().orElse(null);
    if (ground != null) {
      it.units.erallab.hmsrobots.core.geometry.BoundingBox worldFrame = ground.getShape().boundingBox();
      if (miniatureMagnifyRatio != 1f) {
        Point2 center = inWorldFrame.center();
        double w = worldFrame.width();
        double h = worldFrame.height();
        double minX = Math.max(center.x - w / 2d / miniatureMagnifyRatio, worldFrame.min.x);
        worldFrame = it.units.erallab.hmsrobots.core.geometry.BoundingBox.build(
            Point2.build(minX, center.y - h / 2d / miniatureMagnifyRatio),
            Point2.build(minX + w / miniatureMagnifyRatio, center.y + h / 2d / miniatureMagnifyRatio)
        );
      }
      //save original transform
      AffineTransform oAt = g.getTransform();
      //prepare transformation
      double xRatio = graphicsFrame.width() / worldFrame.width();
      double yRatio = graphicsFrame.height() / worldFrame.height();
      double ratio = Math.min(xRatio, yRatio);
      AffineTransform at = new AffineTransform();
      at.translate(graphicsFrame.min.x, graphicsFrame.min.y);
      at.scale(ratio, -ratio);
      at.translate(-worldFrame.min.x, -inWorldFrame.max.y);
      g.setTransform(at);
      //draw ground
      Stroke basicStroke = new BasicStroke(strokeWidth / (float) ratio);
      g.setStroke(basicStroke);
      g.setColor(basicColor);
      MINIATURE_GROUND_DRAWER.draw(ground, null, g);
      //draw in world frame
      Shape rect = new Rectangle2D.Double(inWorldFrame.min.x, inWorldFrame.min.y, inWorldFrame.width(), inWorldFrame.height());
      g.setColor(alphaed(infoColor, 0.25f));
      g.fill(rect);
      g.setColor(infoColor);
      g.draw(rect);
      //restore transform
      g.setTransform(oAt);
    }
  }
*/

  public void draw(double t, List<Snapshot> snapshots, Graphics2D g, BoundingBox graphicsFrame, BoundingBox worldFrame, String... infos) {
    //set clipping area
    g.setClip(
        (int) graphicsFrame.min.x, (int) graphicsFrame.min.y,
        (int) graphicsFrame.width(), (int) graphicsFrame.height()
    );
    //save original transform
    AffineTransform oAt = g.getTransform();
    //prepare transformation
    double xRatio = graphicsFrame.width() / worldFrame.width();
    double yRatio = graphicsFrame.height() / worldFrame.height();
    double ratio = Math.min(xRatio, yRatio);
    AffineTransform at = new AffineTransform();
    at.translate(graphicsFrame.min.x, graphicsFrame.min.y);
    at.scale(ratio, -ratio);
    at.translate(-worldFrame.min.x, -worldFrame.max.y);
    //draw background
    g.setColor(backgroundColor);
    g.fillRect(
        (int) graphicsFrame.min.x, (int) graphicsFrame.min.y,
        (int) graphicsFrame.width(), (int) graphicsFrame.height()
    );
    //draw grid
    g.setTransform(at);
    if (generalRenderingModes.contains(GeneralRenderingMode.GRID_MAJOR) || generalRenderingModes.contains(GeneralRenderingMode.GRID_MINOR)) {
      g.setColor(gridColor);
      g.setStroke(new BasicStroke(1f / (float) ratio));
      double gridSize = computeGridSize(worldFrame.min.x, worldFrame.max.x);
      if (generalRenderingModes.contains(GeneralRenderingMode.GRID_MAJOR)) {
        for (double gridX = Math.floor(worldFrame.min.x / gridSize) * gridSize; gridX < worldFrame.max.x; gridX = gridX + gridSize) {
          g.draw(new Line2D.Double(gridX, worldFrame.min.y, gridX, worldFrame.max.y));
        }
        for (double gridY = Math.floor(worldFrame.min.y / gridSize) * gridSize; gridY < worldFrame.max.y; gridY = gridY + gridSize) {
          g.draw(new Line2D.Double(worldFrame.min.x, gridY, worldFrame.max.x, gridY));
        }
      }
      if (generalRenderingModes.contains(GeneralRenderingMode.GRID_MINOR)) {
        gridSize = gridSize / 5d;
        g.setStroke(new BasicStroke(
            1f / (float) ratio,
            BasicStroke.CAP_BUTT,
            BasicStroke.JOIN_ROUND,
            1.0f,
            new float[]{2f / (float) ratio, 0f, 2f / (float) ratio},
            0f));
        for (double gridX = Math.floor(worldFrame.min.x / gridSize) * gridSize; gridX < worldFrame.max.x; gridX = gridX + gridSize) {
          g.draw(new Line2D.Double(gridX, worldFrame.min.y, gridX, worldFrame.max.y));
        }
        for (double gridY = Math.floor(worldFrame.min.y / gridSize) * gridSize; gridY < worldFrame.max.y; gridY = gridY + gridSize) {
          g.draw(new Line2D.Double(worldFrame.min.x, gridY, worldFrame.max.x, gridY));
        }
      }
    }
    //draw components
    Stroke basicStroke = new BasicStroke(strokeWidth / (float) ratio);
    for (Snapshot snapshot : snapshots) {
      recursivelyDraw(List.of(snapshot), g, basicStroke);
    }
    //restore transform
    g.setTransform(oAt);
    //info
    StringBuilder sb = new StringBuilder();
    if (generalRenderingModes.contains(GeneralRenderingMode.VIEWPORT_INFO)) {
      sb.append((sb.length() > 0) ? " " : "").append(String.format("vp=(%.0f;%.0f)->(%.0f;%.0f)", worldFrame.min.x, worldFrame.min.y, worldFrame.max.x, worldFrame.max.y));
    }
    if (generalRenderingModes.contains(GeneralRenderingMode.TIME_INFO)) {
      sb.append((sb.length() > 0) ? " " : "").append(String.format("t=%.2f", t));
    }
    if (generalRenderingModes.contains(GeneralRenderingMode.ROBOT_CENTERS_INFO)) {
      List<Point2> robotCenters = snapshots.stream()
          .filter(s -> Robot.class.isAssignableFrom(s.getSnapshottableClass()))
          .map(s -> ((BoundingBox) s.getContent()).center())
          .collect(Collectors.toList());
      if (!robotCenters.isEmpty()) {
        sb.append((sb.length() > 0) ? String.format("%n") : "").append("c:");
        for (Point2 center : robotCenters) {
          sb.append(String.format(" (%.0f,%.0f)", center.x, center.y));
        }
      }
    }
    for (String info : infos) {
      sb.append((sb.length() > 0) ? String.format("%n") : "").append(info);
    }
    if (sb.length() > 0) {
      g.setColor(infoColor);
      int relY = 1;
      for (String line : sb.toString().split(String.format("%n"))) {
        g.drawString(line, (int) graphicsFrame.min.x + 1, (int) graphicsFrame.min.y + relY + g.getFontMetrics().getMaxAscent());
        relY = relY + g.getFontMetrics().getMaxAscent() + 1;
      }
    }
    //draw miniature
    /*if (drawMiniature) {
      drawMiniature(
          snapshot,
          g,
          it.units.erallab.hmsrobots.core.geometry.BoundingBox.build(
              Point2.build(
                  graphicsFrame.min.x + graphicsFrame.width() * MINIATURE_REL_BOUNDING_BOX.min.x,
                  graphicsFrame.min.y + graphicsFrame.height() * MINIATURE_REL_BOUNDING_BOX.min.y
              ), Point2.build(
                  graphicsFrame.min.x + graphicsFrame.width() * MINIATURE_REL_BOUNDING_BOX.max.x,
                  graphicsFrame.min.y + graphicsFrame.height() * MINIATURE_REL_BOUNDING_BOX.max.y
              )
          ),
          worldFrame
      );
    }*/
  }

  private static <T> List<T> append(List<T> list, T t) {
    List<T> newList = new ArrayList<>(list);
    newList.add(t);
    return newList;
  }

  private double computeGridSize(double x1, double x2) {
    double gridSize = (x2 - x1) / 10d;
    double exp = Math.floor(Math.log10(gridSize));
    double guess = gridSizes[0];
    double err = Math.abs(gridSize / Math.pow(10d, exp) - guess);
    for (int i = 1; i < gridSizes.length; i++) {
      if (Math.abs(gridSize / Math.pow(10d, exp) - gridSizes[i]) < err) {
        err = Math.abs(gridSize / Math.pow(10d, exp) - gridSizes[i]);
        guess = gridSizes[i];
      }
    }
    gridSize = guess * Math.pow(10d, exp);
    return gridSize;
  }

  private void recursivelyDraw(final List<Snapshot> lineage, final Graphics2D g, Stroke basicStroke) {
    for (Drawer drawer : drawers) {
      g.setStroke(basicStroke);
      g.setColor(basicColor);
      drawer.draw(lineage, g);
    }
    for (Snapshot child : lineage.get(lineage.size() - 1).getChildren()) {
      recursivelyDraw(append(lineage, child), g, basicStroke);
    }
  }

  public static Color linear(final Color c1, final Color c2, final Color c3, float x1, float x2, float x3, float x) {
    if (x < x2) {
      return linear(c1, c2, x1, x2, x);
    }
    return linear(c2, c3, x2, x3, x);
  }

  public static Color linear(final Color c1, final Color c2, float min, float max, float x) {
    x = (x - min) / (max - min);
    x = Float.max(0f, Float.min(1f, x));
    final float r1 = c1.getRed() / 255f;
    final float g1 = c1.getGreen() / 255f;
    final float b1 = c1.getBlue() / 255f;
    final float a1 = c1.getAlpha() / 255f;
    final float r2 = c2.getRed() / 255f;
    final float g2 = c2.getGreen() / 255f;
    final float b2 = c2.getBlue() / 255f;
    final float a2 = c2.getAlpha() / 255f;
    final float r = r1 + (r2 - r1) * x;
    final float g = g1 + (g2 - g1) * x;
    final float b = b1 + (b2 - b1) * x;
    final float a = a1 + (a2 - a1) * x;
    return new Color(r, g, b, a);
  }

  public static Path2D toPath(Poly poly, boolean close) {
    Path2D path = toPath(poly.getVertexes());
    if (close) {
      path.closePath();
    }
    return path;
  }

  public static Path2D toPath(Point2... points) {
    Path2D path = new Path2D.Double();
    path.moveTo(points[0].x, points[0].y);
    for (int i = 1; i < points.length; i++) {
      path.lineTo(points[i].x, points[i].y);
    }
    return path;
  }

  public static Color alphaed(Color color, float alpha) {
    return new Color(
        (float) color.getRed() / 255f,
        (float) color.getGreen() / 255f,
        (float) color.getBlue() / 255f,
        alpha);
  }

}
