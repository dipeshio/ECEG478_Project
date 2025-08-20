/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: Dipesh Bhattarai
 * Date: 4/14/2025
 * Time: 3:42 PM
 *
 * Project: csci205_final_project
 * Package: final_project.Dashboard
 * Class: MainApp
 *
 * Description:
 *
 * ****************************************
 */

package AirForce;

import AirForce.model.CountDownCheck;
import AirForce.model.Sensor;
import AirForce.model.SystemOperation;
import AirForce.model.TimeLine;
import AirForce.view.RocketDashboardApp;
import javafx.application.Application;
import java.io.IOException;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 * Main application class for the rocket dashboard.
 */
public class MainApp {
    public static void main(String[] args) {
        Sensor sensor = new Sensor();
        TimeLine timeLine = new TimeLine();
        CountDownCheck countDownCheck = new CountDownCheck(60 * 1000);
        SystemOperation systemOperation = new SystemOperation();

        // Populate timeline with sample events
        timeLine.addEvent("Launch Prep", "Begin final checks", "T-3h");
        timeLine.addEvent("Ignition", "Engines start", "T-0");
        timeLine.addEvent("Liftoff", "Rocket leaves pad", "T+2s");

        try {
            sensor.readDataFromCSV();
            RocketDashboardApp.setSensorData(sensor.getSensorDataList());
            RocketDashboardApp.setTimeLine(timeLine);
            RocketDashboardApp.setCountDownCheck(countDownCheck);
            RocketDashboardApp.setSystemOperation(systemOperation);
            Application.launch(RocketDashboardApp.class, args);
        } catch (IOException e) {
            System.err.println("Error reading CSV file: " + e.getMessage());
        }
    }
}