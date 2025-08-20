/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 4/20/25
 * Time: 22:34
 *
 * Project: csci205_final_project
 * Package: AirForce.view
 * Class: AccelPlot
 *
 * Description:
 *
 * ****************************************
 */

package AirForce.view;

import AirForce.model.Sensor.SensorData;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import java.util.ArrayList;
import java.util.List;

/**
 * JavaFX application to plot acceleration data (ID vs. AccX, AccY, AccZ).
 */
public class AccelPlotApp extends Application {
    private static List<SensorData> data;

    /**
     * Sets the data to be plotted.
     *
     * @param sensorData List of sensor data
     */
    public static void setData(List<SensorData> sensorData) {
        data = new ArrayList<>(sensorData); // Copy to avoid external modification
    }

    public static void launchApp() {
    }

    @Override
    public void start(Stage stage) {
        if (data == null || data.isEmpty()) {
            System.err.println("No data available for plotting");
            return;
        }

        // Define axes
        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("ID");
        yAxis.setLabel("Acceleration (g)");
        yAxis.setAutoRanging(false);
        yAxis.setLowerBound(-2.5);
        yAxis.setUpperBound(2.5);
        yAxis.setTickUnit(0.5);

        // Create line chart
        LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);
        lineChart.setTitle("Acceleration vs. ID");
        lineChart.setCreateSymbols(false); // Remove markers for cleaner lines

        // Series for AccX
        XYChart.Series<Number, Number> accXSeries = new XYChart.Series<>();
        accXSeries.setName("AccX");

        // Series for AccY
        XYChart.Series<Number, Number> accYSeries = new XYChart.Series<>();
        accYSeries.setName("AccY");

        // Series for AccZ
        XYChart.Series<Number, Number> accZSeries = new XYChart.Series<>();
        accZSeries.setName("AccZ");

        // Populate series with data
        for (SensorData sensorData : data) {
            accXSeries.getData().add(new XYChart.Data<>(sensorData.id(), sensorData.accX()));
            accYSeries.getData().add(new XYChart.Data<>(sensorData.id(), sensorData.accY()));
            accZSeries.getData().add(new XYChart.Data<>(sensorData.id(), sensorData.accZ()));
        }

        // Add series to chart
        lineChart.setCreateSymbols(false);
        lineChart.getData().addAll(accXSeries, accYSeries, accZSeries);

        // Create and show scene
        Scene scene = new Scene(lineChart, 800, 600);
        stage.setScene(scene);
        stage.setTitle("Acceleration Plot");
        stage.show();
    }
}
