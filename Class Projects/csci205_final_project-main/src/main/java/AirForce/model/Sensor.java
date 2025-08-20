/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 4/14/25
 * Time: 1:29 PM
 *
 * Project: csci205_final_project
 * Package: AirForce.model
 * Class: Sensor
 *
 * Description:
 * A class to read and process sensor data from a CSV file (Avionics.csv)
 * and display the extracted data.
 *
 * ****************************************
 */
/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 4/14/25
 * Time: 1:29 PM
 *
 * Project: csci205_final_project
 * Package: AirForce.model
 * Class: Sensor
 *
 * Description:
 * A class to read and process sensor data from a CSV file (Avionics.csv)
 * and display or visualize the extracted data.
 *
 * ****************************************
 */
package AirForce.model;

import AirForce.view.ModelView;
import AirForce.view.RocketDashboardApp;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Represents a sensor that reads and processes data from a CSV file and visualizes it using a polymorphic visualizer.
 */
public class Sensor {
    // Fields to store sensor data and visualizer
    private final List<SensorData> sensorDataList;
    private final ModelView visualizer;

    /**
     * Visualizer for plotting ID vs. AccX, AccY, and AccZ using JavaFX.
     */
    private static class AccelVisualizer implements ModelView {
        @Override
        public void visualize(List<SensorData> data) {
            RocketDashboardApp.setSensorData(data);
        }
    }

    /**
     * Inner record to represent a single row of sensor data.
     */
    public record SensorData(
            int id,
            double accX, double accY, double accZ,
            double gyroX, double gyroY, double gyroZ,
            double yaw, double pitch, double roll,
            double pressure, double temperature,
            double altitude
    ) {
        /**
         * Convenience factory method for creating SensorData with default values for id, pressure, temperature, and altitude.
         */
        public static SensorData createWithDefaults(
                double accX, double accY, double accZ,
                double gyroX, double gyroY, double gyroZ,
                double yaw, double pitch, double roll
        ) {
            return new SensorData(
                    0,          // default id
                    accX, accY, accZ,
                    gyroX, gyroY, gyroZ,
                    yaw, pitch, roll,
                    0.0,        // default pressure
                    0.0,        // default temperature
                    0.0         // default altitude
            );
        }

        @Override
        public String toString() {
            return String.format("ID: %d, AccX: %.2f, AccY: %.2f, AccZ: %.2f, " +
                            "GyroX: %.2f, GyroY: %.2f, GyroZ: %.2f, " +
                            "Yaw: %.2f, Pitch: %.2f, Roll: %.2f, " +
                            "Pressure: %.2f, Temperature: %.2f, Altitude: %.2f",
                    id, accX, accY, accZ, gyroX, gyroY, gyroZ,
                    yaw, pitch, roll, pressure, temperature, altitude);
        }
    }

    /**
     * Constructor initializes the sensor data list and sets default visualizer.
     */
    public Sensor() {
        this.sensorDataList = new ArrayList<>();
        this.visualizer = new AccelVisualizer();
    }

    /**
     * Reads sensor data from the Avionics.csv file in the classpath.
     */
    public void readDataFromCSV() throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                Objects.requireNonNull(Sensor.class.getResourceAsStream("/Avionics.csv"))))) {

            String line;
            boolean isFirstLine = true;
            int lineNumber = 0;

            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (isFirstLine) {
                    isFirstLine = false; // Skip header row
                    continue;
                }

                String[] data = line.split(",");

                if (data.length == 13) {
                    try {
                        SensorData sensorData = getSensorData(data, lineNumber);
                        sensorDataList.add(sensorData);
                    } catch (NumberFormatException e) {
                        System.err.println("Error parsing line " + lineNumber + ": " + line);
                    }
                } else {
                    System.err.println("Invalid data format in line " + lineNumber + ": " + line);
                }
            }
        }
    }

    /**
     * Parses a single row of CSV data into a SensorData object.
     */
    private static SensorData getSensorData(String[] data, int lineNumber) throws NumberFormatException {
        try {
            int id = Integer.parseInt(data[0].trim());
            double accX = Double.parseDouble(data[1].trim());
            double accY = Double.parseDouble(data[2].trim());
            double accZ = Double.parseDouble(data[3].trim());
            double gyroX = Double.parseDouble(data[4].trim());
            double gyroY = Double.parseDouble(data[5].trim());
            double gyroZ = Double.parseDouble(data[6].trim());
            double yaw = Double.parseDouble(data[7].trim());
            double pitch = Double.parseDouble(data[8].trim());
            double roll = Double.parseDouble(data[9].trim());
            double pressure = Double.parseDouble(data[10].trim());
            double temperature = Double.parseDouble(data[11].trim());
            double altitude = Double.parseDouble(data[12].trim());

            return new SensorData(id, accX, accY, accZ, gyroX, gyroY, gyroZ, yaw, pitch, roll,
                    pressure, temperature, altitude);
        } catch (NumberFormatException e) {
            throw new NumberFormatException("Invalid number format in line " + lineNumber);
        }
    }

    /**
     * Returns the list of sensor data (instance method).
     */
    public List<SensorData> getSensorDataList() {
        return new ArrayList<>(sensorDataList);
    }

    /**
     * Static method to align with DashboardController's expectation.
     */
    public static List<SensorData> getSensorData() {
        Sensor sensor = new Sensor();
        try {
            sensor.readDataFromCSV();
        } catch (IOException e) {
            System.err.println("Error reading CSV: " + e.getMessage());
        }
        return sensor.getSensorDataList();
    }

    /**
     * Displays all the sensor data stored in the sensorDataList.
     */
    public void displayData() {
        if (sensorDataList.isEmpty()) {
            System.out.println("No sensor data available.");
            return;
        }
        for (SensorData data : sensorDataList) {
            System.out.println(data);
        }
    }

    /**
     * Visualizes the sensor data using the configured visualizer.
     */
    public void visualize() {
        visualizer.visualize(sensorDataList);
    }
}
