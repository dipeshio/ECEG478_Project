/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 4/20/25
 * Time: 23:01
 *
 * Project: csci205_final_project
 * Package: AirForce.view
 * Class: RocketDashboardApp
 *
 * Description:
 *
 * ****************************************
 */

package AirForce.view;
import AirForce.model.CountDownCheck;
import AirForce.controller.DashboardController;
import AirForce.controller.GUIController;
import AirForce.model.Sensor;
import AirForce.model.SystemOperation;
import AirForce.model.TimeLine;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Screen;
import javafx.stage.Stage;
import java.util.ArrayList;
import java.util.List;

/**
 * The main application class for the Rocket Dashboard in the AirForce application.
 * Extends {@link Application} to provide the JavaFX entry point. Initializes the dashboard UI,
 * loads the FXML layout, and manages static references to sensor data, timeline, countdown check,
 * and system operation.
 */

public class RocketDashboardApp extends Application {
    private static List<Sensor.SensorData> sensorData;     // List of sensor data
    private static TimeLine timeLine;                      // Timeline for events
    private static CountDownCheck countDownCheck;          // Countdown timer
    private static SystemOperation systemOperation;        // System operation state

    /**
     * Sets the sensor data for the application.
     *
     * @param data the list of {@link Sensor.SensorData} to set
     */
    public static void setSensorData(List<Sensor.SensorData> data) {
        sensorData = new ArrayList<>(data);
    }

    /**
     * Sets the timeline for the application.
     *
     * @param tl the {@link TimeLine} to set
     */
    public static void setTimeLine(TimeLine tl) {
        timeLine = tl;
    }

    /**
     * Sets the countdown check for the application.
     *
     * @param cdc the {@link CountDownCheck} to set
     */
    public static void setCountDownCheck(CountDownCheck cdc) {
        countDownCheck = cdc;
    }

    /**
     * Sets the system operation for the application.
     *
     * @param so the {@link SystemOperation} to set
     */
    public static void setSystemOperation(SystemOperation so) {
        systemOperation = so;
    }

    /**
     * Returns the sensor data for the application.
     *
     * @return the list of {@link Sensor.SensorData}
     */
    public static List<Sensor.SensorData> getSensorData() {
        return sensorData;
    }

    /**
     * Returns the timeline for the application.
     *
     * @return the {@link TimeLine}
     */
    public static TimeLine getTimeLine() {
        return timeLine;
    }

    /**
     * Returns the system operation for the application.
     *
     * @return the {@link SystemOperation}
     */
    public static SystemOperation getSystemOperation() {
        return systemOperation;
    }

    /**
     * Returns the countdown check for the application.
     *
     * @return the {@link CountDownCheck}
     */
    public static CountDownCheck getCountDownCheck() {
        return countDownCheck;
    }

    /**
     * Initializes and starts the JavaFX application by loading the FXML layout,
     * setting up the dashboard controller, and configuring the stage.
     *
     * @param stage the primary {@link Stage} for the application
     * @throws Exception if an error occurs during FXML loading or stage setup
     */
    @Override
    public void start(Stage stage) throws Exception {
        // Load FXML
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/dashboard.fxml"));
        AnchorPane root = loader.load();
        DashboardController dashboardController = loader.getController();
        dashboardController.setGUIController(new GUIController(systemOperation, countDownCheck));


        // Set scene size based on screen resolution
        double screenWidth = Screen.getPrimary().getVisualBounds().getWidth() * 0.95; // 80% of screen width
        double screenHeight = Screen.getPrimary().getVisualBounds().getHeight() *0.95; // 80% of screen height
        Scene scene = new Scene(root, screenWidth, screenHeight);


        // Make stage not resizeable
        stage.setScene(scene);
        stage.setTitle("Rocket Dashboard");
        stage.setResizable(true);
        stage.setFullScreen(true);
        stage.show();
    }

    /**
     * The main entry point for the application.
     * Configures system properties for high-DPI scaling and launches the JavaFX application.
     *
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        System.setProperty("prism.allowhidpi", "true"); // Handle high-DPI scaling
        launch(args);
    }
}

