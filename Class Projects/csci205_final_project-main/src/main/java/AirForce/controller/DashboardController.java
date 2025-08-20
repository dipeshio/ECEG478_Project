/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 4/16/2025
 * Time: 3:27 PM
 *
 * Project: csci205_final_project
 * Package: AirForce.controller
 * Class: DashboardController
 *
 * Description:
 *
 * ****************************************
 */


package AirForce.controller;


import AirForce.model.*;
import AirForce.view.PressurePlotApp;
import AirForce.view.RocketDashboardApp;
import AirForce.view.TemperaturePlotApp;
import javafx.animation.AnimationTimer;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.Pane;
import javafx.scene.shape.*;
import javafx.scene.text.Text;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.Objects;
import java.util.ResourceBundle;
import java.util.concurrent.ConcurrentLinkedQueue;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.paint.Color;


/**
 * Controller class for the rocket dashboard user interface in the AirForce application.
 * Manages the display and interaction of sensor data, weather information, gauges, graphs,
 * and live video feed using JavaFX components. Implements {@link Initializable} to initialize
 * the UI components and data upon loading.
 */


public class DashboardController implements Initializable {
    @FXML private TextField accXText;
    @FXML private TextField accYText;
    @FXML private TextField accZText;
    @FXML private LineChart<String, Number> accelGraph;
    @FXML private Button bmpButton;
    @FXML private Button gpsButton;
    @FXML private TextField gyroXText;
    @FXML private LineChart<String, Number> gyroGraph;
    @FXML private TextField gyroYText;
    @FXML private TextField gyroZText;
    @FXML private Button mpuButton;
    @FXML private LineChart<String, Number> pitchGraph;
    @FXML private TextField pitchText;
    @FXML private Button powerOff;
    @FXML private Button powerOn;
    @FXML private Button recordButton;
    @FXML private TextField rollText;
    @FXML private Button tlmButton;
    @FXML private TextField yawText;
    @FXML private Label countdownLabel;
    @FXML private ListView<String> timelineView;
    @FXML private MediaView mediaPane;
    @FXML private TextField standByText, deployedText, launchText;
    @FXML private Pane logoPane;
//    @FXML private TextField pressureLabel1;
//    @FXML private TextField tempLabel1;
    @FXML private Pane pressureGaugePane;
    @FXML private Pane tempGaugePane;
    @FXML private Text temperatureText;
    @FXML private Text sunsetText;
    @FXML private Text latitudeText;
    @FXML private Text longitudeText;
    @FXML private Text windSpeedText;
    @FXML private Text visibilityText;
    @FXML private Text locationText;
    @FXML private LineChart<String, Number> altitudeGraph;
    @FXML private ImageView mapImageView;
    @FXML private Text voltageText;
    @FXML private ImageView trajectoryMapImageView;
    @FXML private Text trajectoryLocationText;
    @FXML private Text trajectoryLatitudeText;
    @FXML private Text trajectoryLongitudeText;
    @FXML private Pane logoBack2;
    @FXML private TextField realTimePressureText;
    @FXML private TextField realTimeTempText;


    private GUIController guiController;     // Controller for managing system operations
    private PressurePlotApp pressurePlotApp; // Application for plotting pressure data
    private TemperaturePlotApp temperaturePlotApp; // Application for plotting temperature data
    private int sensorDataIndex;             // Index for iterating through sensor data
    private Location location;               // Represents the current geographical location
    private static final String GOOGLE_MAPS_API_KEY = "AIzaSyCyEVAMiQ4o2DYwU1ZnZJ_kbiJyPWpF0xs"; // Google Maps API key
    private final ConcurrentLinkedQueue<Number> accXDataQ = new ConcurrentLinkedQueue<>(); // Queue for X-axis acceleration data
    private final ConcurrentLinkedQueue<Number> accYDataQ = new ConcurrentLinkedQueue<>(); // Queue for Y-axis acceleration data
    private final ConcurrentLinkedQueue<Number> accZDataQ = new ConcurrentLinkedQueue<>(); // Queue for Z-axis acceleration data
    private final ConcurrentLinkedQueue<Number> gyroXDataQ = new ConcurrentLinkedQueue<>(); // Queue for X-axis gyroscope data
    private final ConcurrentLinkedQueue<Number> gyroYDataQ = new ConcurrentLinkedQueue<>(); // Queue for Y-axis gyroscope data
    private final ConcurrentLinkedQueue<Number> gyroZDataQ = new ConcurrentLinkedQueue<>(); // Queue for Z-axis gyroscope data
    private final ConcurrentLinkedQueue<Number> yawDataQ = new ConcurrentLinkedQueue<>(); // Queue for yaw data
    private final ConcurrentLinkedQueue<Number> rollDataQ = new ConcurrentLinkedQueue<>(); // Queue for roll data
    private final ConcurrentLinkedQueue<Number> pitchDataQ = new ConcurrentLinkedQueue<>(); // Queue for pitch data
    private final ConcurrentLinkedQueue<Number> tempDataQ = new ConcurrentLinkedQueue<>(); // Queue for temperature data
    private final ConcurrentLinkedQueue<Number> altitudeDataQ = new ConcurrentLinkedQueue<>(); // Queue for altitude data
    private final ConcurrentLinkedQueue<Number> pressureDataQ = new ConcurrentLinkedQueue<>(); // Queue for pressure data
    private final XYChart.Series<String, Number> accXSeries = new XYChart.Series<>(); // Series for X-axis acceleration graph
    private final XYChart.Series<String, Number> accYSeries = new XYChart.Series<>(); // Series for Y-axis acceleration graph
    private final XYChart.Series<String, Number> accZSeries = new XYChart.Series<>(); // Series for Z-axis acceleration graph
    private final XYChart.Series<String, Number> gyroXSeries = new XYChart.Series<>(); // Series for X-axis gyroscope graph
    private final XYChart.Series<String, Number> gyroYSeries = new XYChart.Series<>(); // Series for Y-axis gyroscope graph
    private final XYChart.Series<String, Number> gyroZSeries = new XYChart.Series<>(); // Series for Z-axis gyroscope graph
    private final XYChart.Series<String, Number> pitchSeries = new XYChart.Series<>(); // Series for pitch graph
    private final XYChart.Series<String, Number> yawSeries = new XYChart.Series<>(); // Series for yaw graph
    private final XYChart.Series<String, Number> rollSeries = new XYChart.Series<>(); // Series for roll graph
    private final XYChart.Series<String, Number> altitudeSeries = new XYChart.Series<>(); // Series for altitude graph
    private int xSeriesData = 0; // Counter for graph data points
    private static final int MAX_DATA_POINTS = 20; // Maximum number of data
    private MediaPlayer mediaPlayer; // Media player for live video
    private LiveFeed liveFeed; // Manages live video feed
    private boolean isVideoPlaying = false; // Indicates if video is currently
    private boolean isTLM20 = false; // Indicates if telemetry rate is TLM20
    private boolean isMPUOn = true; // Indicates if MPU6050 sensor is on
    private boolean isBMPOn = true; // Indicates if BMP180 sensor is on
    private boolean isPowerOn = true; // Indicates if system power is on
    private static final long UPDATE_INTERVAL_TLM_500 = 200_000_000; // Update interval for TLM 500 (200ms)
    private static final long UPDATE_INTERVAL_TLM_20 = 1000_000_000; // Update interval for TLM 500 (1s)


    /**
     * Sets the GUI controller for managing system operations.
     *
     * @param controller the {@link GUIController} instance
     */
    public void setGUIController(GUIController controller) {
        this.guiController = controller;
    }


    /**
     * Initializes the controller after its root element has been processed.
     * Sets up UI components, sensor data, graphs, gauges, weather data, and starts the animation timer.
     *
     * @param location  the location used to resolve relative paths for the root object
     * @param resources the resources used to localize the root object
     */
    @Override
    public void initialize(URL location, ResourceBundle resources) {
        guiController = new GUIController(
                RocketDashboardApp.getSystemOperation(),
                RocketDashboardApp.getCountDownCheck()
        );


        accXSeries.setName("AccX");
        accYSeries.setName("AccY");
        accZSeries.setName("AccZ");
        gyroXSeries.setName("GyroX");
        gyroYSeries.setName("GyroY");
        gyroZSeries.setName("GyroZ");
        pitchSeries.setName("Pitch");
        yawSeries.setName("Yaw");
        rollSeries.setName("Roll");


        accelGraph.getData().addAll(accXSeries, accYSeries, accZSeries);
        accelGraph.setCreateSymbols(false);


        gyroGraph.getData().addAll(gyroXSeries, gyroYSeries, gyroZSeries);
        gyroGraph.setCreateSymbols(false);


        pitchGraph.getData().addAll(pitchSeries, yawSeries, rollSeries);
        pitchGraph.setCreateSymbols(false);


        // Initialize altitude graph
        altitudeGraph.getData().add(altitudeSeries);
        altitudeGraph.setCreateSymbols(false);
        altitudeGraph.setLegendVisible(false);


        liveFeed = new LiveFeed(mediaPane);
        fetchWeather();


        // Initialize map with default image
        if (mapImageView != null) {
            String defaultMapUrl = "https://maps.googleapis.com/maps/api/staticmap?center=0,0&zoom=12&size=267x190&markers=color:red|0,0&key=" + GOOGLE_MAPS_API_KEY;
            try {
                mapImageView.setImage(new Image(defaultMapUrl));
            } catch (Exception e) {
                System.err.println("Failed to load default map image: " + e.getMessage());
            }
        }


        // Initialize map for tracjectoryPane with default image
        if (trajectoryMapImageView != null) {
            String defaultMapUrl = "https://maps.googleapis.com/maps/api/staticmap?center=0,0&zoom=12&size=267x190&markers=color:red|0,0&key=" + GOOGLE_MAPS_API_KEY;
            try {
                trajectoryMapImageView.setImage(new Image(defaultMapUrl));
            } catch (Exception e) {
                System.err.println("Failed to load default map image for tracjectoryPane: " + e.getMessage());
            }
        }


        // Initialize PlotApps first
        pressurePlotApp = new PressurePlotApp(realTimePressureText, pressureGaugePane, pressureDataQ);
        temperaturePlotApp = new TemperaturePlotApp(realTimeTempText, tempGaugePane, tempDataQ);


        // Initialize gauges
        createPressureGauge();
        createTempGauge();


        // Initialize Location
        this.location = new Location("Lewisburg,PA,US");




        // Load sensor data
        List<Sensor.SensorData> sensorData = RocketDashboardApp.getSensorData();
        if (sensorData != null && !sensorData.isEmpty()) {
            getSensorData(sensorData);
        } else {
            System.err.println("No sensor data available.");
        }


        // Initialize timeline
        TimeLine timeLine = RocketDashboardApp.getTimeLine();
        if (timeLine != null && timelineView != null) {
            for (TimeLine.TimeLineEvent event : timeLine.getEvents()) {
                timelineView.getItems().add(
                        event.timeMarker() + " - " + event.title() + ": " + event.description()
                );
            }
        }

// Load logo
        try {
            Image logoImage = new Image(Objects.requireNonNull(getClass().getResourceAsStream("/Avakas.png")));
            ImageView logoView = new ImageView(logoImage);
            logoView.setFitWidth(200);
            logoView.setFitHeight(200);
            logoView.setPreserveRatio(true);

            // Set clipping region to ensure the image stays within the pane
            Rectangle clip = new Rectangle(logoBack2.getWidth(), logoBack2.getHeight());
            logoBack2.setClip(clip);

            logoBack2.getChildren().add(logoView);
        } catch (Exception e) {
            System.err.println("Error loading logo image: " + e.getMessage());
            e.printStackTrace();
        }


        // Start countdown
        if (countdownLabel != null) {
            guiController.startCountdownTimer(countdownLabel);
            countdownLabel.textProperty().addListener((obs, oldVal, newVal) -> {
            });
        }


        // Button actions
        powerOn.setOnAction(e -> {
            guiController.handlePowerToggle(createToggleButton("Power ON", true));
            powerOn.setText("Power ON");
            powerOff.setText("Power OFF");
            guiController.startCountdownTimer(countdownLabel);
        });
        powerOff.setOnAction(e -> {
            guiController.handlePowerToggle(createToggleButton("Power OFF", false));
            powerOn.setText("Power ON");
            powerOff.setText("Power OFF");
            guiController.stopCountdownTimer();
            guiController.resetCountdownTimer(countdownLabel);
            if (mediaPlayer != null && isVideoPlaying) {
                mediaPlayer.stop();
                isVideoPlaying = false;
            }
        });
        mpuButton.setOnAction(e -> {
            isMPUOn = !isMPUOn;
            guiController.handleMPUToggle(createToggleButton(mpuButton.getText(), isMPUOn));
            mpuButton.setText(isMPUOn ? "MPU6050 ON" : "MPU6050 OFF");
        });
        bmpButton.setOnAction(e -> {
            isBMPOn = !isBMPOn;
            guiController.handleBMPToggle(createToggleButton(bmpButton.getText(), isBMPOn));
            bmpButton.setText(isBMPOn ? "BMP 180 ON" : "BMP 180 OFF");
        });
        tlmButton.setOnAction(e -> {
            isTLM20 = !isTLM20;
            tlmButton.setText(isTLM20 ? "TLM 20" : "TLM 500");
            guiController.handleTLMToggle(createToggleButton(tlmButton.getText(), isTLM20));
        });
        gpsButton.setOnAction(e -> {
            guiController.handleGPSToggle(createToggleButton(gpsButton.getText(), gpsButton.getText().contains("ON")));
            gpsButton.setText(gpsButton.getText().contains("ON") ? "GPS OFF" : "GPS ON");
        });
        recordButton.setOnAction(e -> recordButton.setText(recordButton.getText().equals("Record") ? "Stop" : "Record"));


        // Initialize power state
        isPowerOn = guiController.isPowerOn();
        updatePowerUI();


        if (powerOn != null) {
            powerOn.setOnAction(e -> {
                System.out.println("Power On button clicked.");
                guiController.powerOn();
                isPowerOn = true;
                updatePowerUI();
                guiController.startCountdownTimer(countdownLabel);
            });
        } else {
            System.err.println("powerOn is null after FXML injection.");
        }


        if (powerOff != null) {
            powerOff.setOnAction(e -> {
                System.out.println("Power Off button clicked.");
                guiController.powerOff();
                isPowerOn = false;
                updatePowerUI();
                guiController.stopCountdownTimer();
                guiController.resetCountdownTimer(countdownLabel);
                if (mediaPlayer != null && isVideoPlaying) {
                    mediaPlayer.stop();
                    isVideoPlaying = false;
                }
            });
        } else {
            System.err.println("powerOff is null after FXML injection.");
        }


        // Animation timer
        new AnimationTimer() {
            private long lastUpdate = 0;


            @Override
            public void handle(long now) {
                long updateInterval = isTLM20 ? UPDATE_INTERVAL_TLM_20 : UPDATE_INTERVAL_TLM_500;
                if (now - lastUpdate >= updateInterval) {
                    assert sensorData != null;
                    if (!sensorData.isEmpty()) {
                        Sensor.SensorData currentData = sensorData.get(sensorDataIndex);
                        pressureDataQ.add(currentData.pressure());
                        tempDataQ.add(currentData.temperature());


                        // Move to next data point, loop back if at end
                        sensorDataIndex = (sensorDataIndex + 1) % sensorData.size();
                    } else {
                        // Fallback if no data
                        pressureDataQ.add(0.0);
                        tempDataQ.add(0.0);
                    }


                    addDataToSeries();
                    updateSensorTextFieldsDynamically();
                    pressurePlotApp.addDataToSeries(isBMPOn);
                    temperaturePlotApp.addDataToSeries(isBMPOn);
                    updateButtonStates();
                    updateLocationTextFields();
                    lastUpdate = now;
                }
            }
        }.start();
    }


    /**
     * Retrieves and processes sensor data for display.
     *
     * @param sensorData list of {@link Sensor.SensorData} objects
     */


    private void getSensorData(List<Sensor.SensorData> sensorData) {
        updateSensorData(sensorData);
        updateSensorTextFields(sensorData.getLast());
    }


    /**
     * Creates a pressure gauge visualization in the pressure gauge pane.
     */
    private void createPressureGauge() {
        Pane pane = pressureGaugePane;
        pane.getChildren().clear();
        double centerX = 60.0;
        double centerY = 60.0;
        double radius = 50.0;
        double innerRadius = radius - 10;


        // Outer ring
        Arc outerRing = new Arc(centerX, centerY, radius, radius, 0, 360);
        outerRing.setType(ArcType.OPEN);
        outerRing.setFill(null);
        outerRing.setStroke(Color.GRAY);
        outerRing.setStrokeWidth(4.0);
        pane.getChildren().add(outerRing);


        // Colored segments
        Arc yellowArc = new Arc(centerX, centerY, innerRadius, innerRadius, 160, 80); // 120° to 180°
        yellowArc.setType(ArcType.ROUND);
        yellowArc.setFill(Color.YELLOW);
        pane.getChildren().add(yellowArc);

        Arc greenArc = new Arc(centerX, centerY, innerRadius, innerRadius, 20, 140); // -60° to 0°
        greenArc.setType(ArcType.ROUND);
        greenArc.setFill(Color.GREEN);
        pane.getChildren().add(greenArc);

        Arc redArc = new Arc(centerX, centerY, innerRadius, innerRadius, -60, 80); // 0° to 120°
        redArc.setType(ArcType.ROUND);
        redArc.setFill(Color.RED);
        pane.getChildren().add(redArc);


        // Tick marks (every 15°)
        for (int i = 0; i <= 20; i++) {
            double angle = 120 + (i * 15);
            getRadian(pane, centerX, centerY, innerRadius, angle);
        }


        // Center pivot
        Circle pivot = new Circle(centerX, centerY, 5);
        pivot.setFill(Color.BLACK);
        pane.getChildren().add(pivot);


        // Needle
        Arc needle = new Arc(centerX, centerY, 5.0, innerRadius - 15, 90, 10);
        needle.setType(ArcType.ROUND);
        needle.setFill(Color.BLACK);
        pane.getChildren().add(needle);


        if (pressurePlotApp != null) {
            pressurePlotApp.setNeedle(needle);
        }
    }


    /**
     * Draws tick marks on a gauge at specified angles.
     *
     * @param pane     the pane to draw on
     * @param centerX  the x-coordinate of the gauge center
     * @param centerY  the y-coordinate of the gauge center
     * @param innerRadius the radius for tick marks
     * @param angle    the angle for the tick mark (in degrees)
     */


    private void getRadian(Pane pane, double centerX, double centerY, double innerRadius, double angle) {
        double rad = Math.toRadians(angle);
        double tickX1 = centerX + (innerRadius - 5) * Math.cos(rad);
        double tickY1 = centerY + (innerRadius - 5) * Math.sin(rad);
        double tickX2 = centerX + (innerRadius + 5) * Math.cos(rad);
        double tickY2 = centerY + (innerRadius + 5) * Math.sin(rad);
        Line tick = new Line(tickX1, tickY1, tickX2, tickY2);
        tick.setStroke(Color.BLACK);
        pane.getChildren().add(tick);
    }


    /**
     * Creates a temperature gauge visualization in the temperature gauge pane.
     */
    private void createTempGauge() {
        Pane pane = tempGaugePane;
        pane.getChildren().clear();
        double centerX = 60.0;
        double centerY = 60.0;
        double radius = 50.0;
        double innerRadius = radius - 10;


        // Outer ring
        Arc outerRing = new Arc(centerX, centerY, radius, radius, 0, 180);
        outerRing.setType(ArcType.OPEN);
        outerRing.setFill(null);
        outerRing.setStroke(Color.GRAY);
        outerRing.setStrokeWidth(4.0);
        pane.getChildren().add(outerRing);


        // Colored segments
        Arc yellowArc = new Arc(centerX, centerY, innerRadius, innerRadius, 120, 60); // 120° to 180°
        yellowArc.setType(ArcType.ROUND);
        yellowArc.setFill(Color.YELLOW);
        pane.getChildren().add(yellowArc);

        Arc greenArc = new Arc(centerX, centerY, innerRadius, innerRadius, 60, 60); // -60° to 0°
        greenArc.setType(ArcType.ROUND);
        greenArc.setFill(Color.GREEN);
        pane.getChildren().add(greenArc);

        Arc redArc = new Arc(centerX, centerY, innerRadius, innerRadius, 0, 60); // 0° to 120°
        redArc.setType(ArcType.ROUND);
        redArc.setFill(Color.RED);
        pane.getChildren().add(redArc);


        // Tick marks (every 15°)
        for (int i = 0; i <= 12; i++) {
            double angle = 180 + (i * 15);
            getRadian(pane, centerX, centerY, innerRadius, angle);
        }


        // Center pivot
        Circle pivot = new Circle(centerX, centerY, 5);
        pivot.setFill(Color.BLACK);
        pane.getChildren().add(pivot);


        // Needle
        Arc needle = new Arc(centerX, centerY, 5.0, innerRadius - 15, 90, 10);
        needle.setType(ArcType.ROUND);
        needle.setFill(Color.BLACK);
        pane.getChildren().add(needle);


        if (temperaturePlotApp != null) {
            temperaturePlotApp.setNeedle(needle);
        }
    }


    // In DashboardController.java
    private void updateButtonStates() {
        String[] timeParts = countdownLabel.getText().split(":");
        int hours = Integer.parseInt(timeParts[0]);
        int minutes = Integer.parseInt(timeParts[1]);
        int seconds = Integer.parseInt(timeParts[2]);
        int totalSeconds = hours * 3600 + minutes * 60 + seconds;

        if (standByText != null) {
            standByText.setStyle("-fx-background-color: #ff6a00;");
        }
        if (deployedText != null) {
            deployedText.setStyle("-fx-background-color: #ff6a00;");
        }
        if (launchText != null) {
            launchText.setStyle("-fx-background-color: #ff6a00;");
        }


        boolean shouldVideoPlay = false;


        if (totalSeconds > 40) {
            if (standByText != null) {
                standByText.setStyle("-fx-background-color: #00FF00;");
            }
        } else if (totalSeconds < 30) {
            if (deployedText != null) {
                deployedText.setStyle("-fx-background-color: #00FF00;");
            }
            if (totalSeconds < 20) {
                if (launchText != null) {
                    launchText.setStyle("-fx-background-color: #00FF00;");
                }
                shouldVideoPlay = true;
            }
        }


        if (shouldVideoPlay && !liveFeed.isVideoPlaying()) {
            liveFeed.play();
        } else if (!shouldVideoPlay && liveFeed.isVideoPlaying()) {
            liveFeed.stop();
        }
    }


    /**
     * Fetches weather data for the specified location and updates UI components.
     */
    private void fetchWeather() {
        try {
            WeatherAPI weatherService = new WeatherAPI();
            String weatherData = weatherService.fetchWeatherData("Lewisburg,PA,US");
            // Extract weather data
            String displayLocation = extractLocation(weatherData);
            String temperature = extractTemperature(weatherData);
            String sunset = extractSunset(weatherData);
            String windSpeed = extractWindSpeed(weatherData);
            String visibility = extractVisibility(weatherData);
            double latitude = extractLatitude(weatherData);
            double longitude = extractLongitude(weatherData);


            // Update recoveryPane
            if (locationText != null) locationText.setText("Location: " + displayLocation);
            if (latitudeText != null) latitudeText.setText("Latitude: " + latitude);
            if (longitudeText != null) longitudeText.setText("Longitude: " + longitude);
            if (temperatureText != null) temperatureText.setText("Temperature: " + temperature);
            if (sunsetText != null) sunsetText.setText("Sunset: " + sunset);
            if (windSpeedText != null) windSpeedText.setText("Wind Speed: " + windSpeed);
            if (visibilityText != null) visibilityText.setText("Visibility: " + visibility);


            // Update tracjectoryPane
            if (trajectoryLocationText != null) trajectoryLocationText.setText("Location: " + displayLocation);
            if (trajectoryLatitudeText != null) trajectoryLatitudeText.setText("Latitude: " + latitude);
            if (trajectoryLongitudeText != null) trajectoryLongitudeText.setText("Longitude: " + longitude);


            // Update map for both recoveryPane and tracjectoryPane
            String mapUrl = "https://maps.googleapis.com/maps/api/staticmap?center=" + latitude + "," + longitude + "&zoom=17&size=267x190&maptype=satellite&markers=color:red|" + latitude + "," + longitude + "&key=" + GOOGLE_MAPS_API_KEY;
            try {
                Image mapImage = new Image(mapUrl, true);
                if (mapImageView != null) {
                    mapImageView.setImage(null);
                    mapImageView.setImage(mapImage);
                    mapImageView.setVisible(false);
                    mapImageView.setVisible(true);
                    System.out.println("Map image loaded successfully for recoveryPane from: " + mapUrl);
                }
                if (trajectoryMapImageView != null) {
                    trajectoryMapImageView.setImage(null);
                    trajectoryMapImageView.setImage(mapImage);
                    trajectoryMapImageView.setVisible(false);
                    trajectoryMapImageView.setVisible(true);
                    System.out.println("Map image loaded successfully for tracjectoryPane from: " + mapUrl);
                }
            } catch (Exception e) {
                System.err.println("Failed to load map image from: " + mapUrl);
                e.printStackTrace();
                // Fallback to temp file
                try {
                    URL url = new URL(mapUrl);
                    java.awt.image.BufferedImage img = javax.imageio.ImageIO.read(url);
                    java.io.File tempFile = java.io.File.createTempFile("map", ".png");
                    javax.imageio.ImageIO.write(img, "png", tempFile);
                    Image tempImage = new Image(tempFile.toURI().toString());
                    if (mapImageView != null) {
                        mapImageView.setImage(null);
                        mapImageView.setImage(tempImage);
                        mapImageView.setVisible(false);
                        mapImageView.setVisible(true);
                    }
                    if (trajectoryMapImageView != null) {
                        trajectoryMapImageView.setImage(null);
                        trajectoryMapImageView.setImage(tempImage);
                        trajectoryMapImageView.setVisible(false);
                        trajectoryMapImageView.setVisible(true);
                    }
                    tempFile.deleteOnExit();
                    System.out.println("Map image loaded from temp file: " + tempFile.getAbsolutePath());
                } catch (Exception ex) {
                    System.err.println("Failed to load map image via temp file: " + ex.getMessage());
                    ex.printStackTrace();
                    // Fallback to placeholder image
                    try {
                        getImage();
                        System.out.println("Placeholder image loaded.");
                    } catch (Exception placeholderEx) {
                        System.err.println("Failed to load placeholder image: " + placeholderEx.getMessage());
                        placeholderEx.printStackTrace();
                    }
                }
            }
        } catch (IOException | InterruptedException e) {
            // Update recoveryPane
            if (locationText != null) locationText.setText("Location: Error fetching data");
            if (latitudeText != null) latitudeText.setText("Latitude: Error fetching data");
            if (longitudeText != null) longitudeText.setText("Longitude: Error fetching data");
            if (temperatureText != null) temperatureText.setText("Temperature: Error fetching data");
            if (sunsetText != null) sunsetText.setText("Sunset: Error fetching data");
            if (windSpeedText != null) windSpeedText.setText("Wind Speed: Error fetching data");
            if (visibilityText != null) visibilityText.setText("Visibility: Error fetching data");


            // Update tracjectoryPane
            if (trajectoryLocationText != null) trajectoryLocationText.setText("Location: Error fetching data");
            if (trajectoryLatitudeText != null) trajectoryLatitudeText.setText("Latitude: Error fetching data");
            if (trajectoryLongitudeText != null) trajectoryLongitudeText.setText("Longitude: Error fetching data");


            // Fallback map for both panes
            String defaultMapUrl = "https://maps.googleapis.com/maps/api/staticmap?center=0,0&zoom=12&size=267x190&markers=color:red|0,0&key=" + GOOGLE_MAPS_API_KEY;
            try {
                Image defaultImage = new Image(defaultMapUrl, true);
                if (mapImageView != null) {
                    mapImageView.setImage(null);
                    mapImageView.setImage(defaultImage);
                    mapImageView.setVisible(false);
                    mapImageView.setVisible(true);
                }
                if (trajectoryMapImageView != null) {
                    trajectoryMapImageView.setImage(null);
                    trajectoryMapImageView.setImage(defaultImage);
                    trajectoryMapImageView.setVisible(false);
                    trajectoryMapImageView.setVisible(true);
                }
            } catch (Exception ex) {
                System.err.println("Failed to load default map image: " + ex.getMessage());
                ex.printStackTrace();
                try {
                    getImage();
                    System.out.println("Placeholder image loaded in error case.");
                } catch (Exception placeholderEx) {
                    System.err.println("Failed to load placeholder image in error case: " + placeholderEx.getMessage());
                    placeholderEx.printStackTrace();
                }
            }
            e.printStackTrace();
        }
    }


    /**
     * Loads a placeholder map image when the actual map fails to load.
     */


    private void getImage() {
        Image placeholderImage = new Image(Objects.requireNonNull(getClass().getResourceAsStream("/placeholder_map.png")));
        if (mapImageView != null) {
            mapImageView.setImage(placeholderImage);
            locationText.setText("Location: Map failed to load (showing placeholder)");
            mapImageView.setVisible(false);
            mapImageView.setVisible(true);
        }
        if (trajectoryMapImageView != null) {
            trajectoryMapImageView.setImage(placeholderImage);
            trajectoryLocationText.setText("Location: Map failed to load (showing placeholder)");
            trajectoryMapImageView.setVisible(false);
            trajectoryMapImageView.setVisible(true);
        }
    }


    /**
     * Extracts the location name from weather data.
     *
     * @param weatherData the raw weather data string
     * @return the location name, or "Not available" if extraction fails
     */


    private String extractLocation(String weatherData) {
        try {
            String locationMarker = "Location: ";
            int locationStart = weatherData.indexOf(locationMarker) + locationMarker.length();
            int locationEnd = weatherData.indexOf("\n", locationStart);
            if (locationStart < locationMarker.length() || locationEnd == -1) {
                return "Not available";
            }
            return weatherData.substring(locationStart, locationEnd).trim();
        } catch (Exception e) {
            return "Not available";
        }
    }


    /**
     * Extracts the latitude from weather data.
     *
     * @param weatherData the raw weather data string
     * @return the latitude, or 0.0 if extraction fails
     */
    private double extractLatitude(String weatherData) {
        try {
            String latMarker = "Latitude: ";
            int latStart = weatherData.indexOf(latMarker) + latMarker.length();
            int latEnd = weatherData.indexOf("\n", latStart);
            if (latStart < latMarker.length() || latEnd == -1) {
                return 0.0;
            }
            return Double.parseDouble(weatherData.substring(latStart, latEnd).trim());
        } catch (Exception e) {
            return 0.0;
        }
    }


    /**
     * Extracts the longitude from weather data.
     *
     * @param weatherData the raw weather data string
     * @return the longitude, or 0.0 if extraction fails
     */
    private double extractLongitude(String weatherData) {
        try {
            String lonMarker = "Longitude: ";
            int lonStart = weatherData.indexOf(lonMarker) + lonMarker.length();
            int lonEnd = weatherData.indexOf("\n", lonStart);
            if (lonStart < lonMarker.length() || lonEnd == -1) {
                return 0.0;
            }
            return Double.parseDouble(weatherData.substring(lonStart, lonEnd).trim());
        } catch (Exception e) {
            return 0.0;
        }
    }


    /**
     * Extracts the wind speed from weather data.
     *
     * @param weatherData the raw weather data string
     * @return the wind speed, or "Not available" if extraction fails
     */
    private String extractWindSpeed(String weatherData) {
        try {
            String windSpeedMarker = "Wind Speed: ";
            int windSpeedStart = weatherData.indexOf(windSpeedMarker) + windSpeedMarker.length();
            int windSpeedEnd = weatherData.indexOf(" m/s", windSpeedStart);
            if (windSpeedStart < windSpeedMarker.length() || windSpeedEnd == -1) {
                return "Not available";
            }
            return weatherData.substring(windSpeedStart, windSpeedEnd + 4).trim();
        } catch (Exception e) {
            return "Not available";
        }
    }


    /**
     * Extracts the visibility from weather data.
     *
     * @param weatherData the raw weather data string
     * @return the visibility, or "Not available" if extraction fails
     */
    private String extractVisibility(String weatherData) {
        try {
            String visibilityMarker = "Visibility: ";
            int visibilityStart = weatherData.indexOf(visibilityMarker) + visibilityMarker.length();
            int visibilityEnd = weatherData.indexOf(" m", visibilityStart);
            if (visibilityStart < visibilityMarker.length() || visibilityEnd == -1) {
                return "Not available";
            }
            return weatherData.substring(visibilityStart, visibilityEnd + 2).trim();
        } catch (Exception e) {
            return "Not available";
        }
    }


    /**
     * Extracts the temperature from weather data.
     *
     * @param weatherData the raw weather data string
     * @return the temperature, or "Not available" if extraction fails
     */
    private String extractTemperature(String weatherData) {
        try {
            String tempMarker = "Temperature: ";
            int tempStart = weatherData.indexOf(tempMarker) + tempMarker.length();
            int tempEnd = weatherData.indexOf("°C", tempStart);
            if (tempStart < tempMarker.length() || tempEnd == -1) {
                return "Not available";
            }
            return weatherData.substring(tempStart, tempEnd + 2).trim();
        } catch (Exception e) {
            return "Not available";
        }
    }


    /**
     * Extracts the sunset time from weather data.
     *
     * @param weatherData the raw weather data string
     * @return the sunset time, or "Not available" if extraction fails
     */
    private String extractSunset(String weatherData) {
        try {
            String sunsetMarker = "Sunset: ";
            int sunsetStart = weatherData.indexOf(sunsetMarker) + sunsetMarker.length();
            int sunsetEnd = weatherData.length(); // Sunset is the last line
            if (sunsetStart < sunsetMarker.length()) {
                return "Not available";
            }
            return weatherData.substring(sunsetStart, sunsetEnd).trim();
        } catch (Exception e) {
            return "Not available";
        }
    }


    /**
     * Updates sensor text fields with the provided sensor data.
     *
     * @param data the {@link Sensor.SensorData} object containing sensor values
     */
    private void updateSensorTextFields(Sensor.SensorData data) {
        if (accXText != null) accXText.setText(isMPUOn ? String.format("AccX: %.2f", data.accX()) : "AccX: OFF");
        if (accYText != null) accYText.setText(isMPUOn ? String.format("AccY: %.2f", data.accY()) : "AccY: OFF");
        if (accZText != null) accZText.setText(isMPUOn ? String.format("AccZ: %.2f", data.accZ()) : "AccZ: OFF");
        if (gyroXText != null) gyroXText.setText(isMPUOn ? String.format("GyroX: %.2f", data.gyroX()) : "GyroX: OFF");
        if (gyroYText != null) gyroYText.setText(isMPUOn ? String.format("GyroY: %.2f", data.gyroY()) : "GyroY: OFF");
        if (gyroZText != null) gyroZText.setText(isMPUOn ? String.format("GyroZ: %.2f", data.gyroZ()) : "GyroZ: OFF");
        if (yawText != null) yawText.setText(isMPUOn ? String.format("Yaw: %.2f", data.yaw()) : "Yaw: OFF");
        if (rollText != null) rollText.setText(isMPUOn ? String.format("Roll: %.2f", data.roll()) : "Roll: OFF");
        if (pitchText != null) pitchText.setText(isMPUOn ? String.format("Pitch: %.2f", data.pitch()) : "Pitch: OFF");
    }


    /**
     * Dynamically updates sensor text fields based on the latest queued data.
     */
    private void updateSensorTextFieldsDynamically() {
        if (!accXDataQ.isEmpty()) {
            if (isMPUOn) {
                double accX = accXDataQ.peek().doubleValue();
                assert accYDataQ.peek() != null;
                double accY = accYDataQ.peek().doubleValue();
                assert accZDataQ.peek() != null;
                double accZ = accZDataQ.peek().doubleValue();
                assert gyroXDataQ.peek() != null;
                double gyroX = gyroXDataQ.peek().doubleValue();
                assert gyroYDataQ.peek() != null;
                double gyroY = gyroYDataQ.peek().doubleValue();
                assert gyroZDataQ.peek() != null;
                double gyroZ = gyroZDataQ.peek().doubleValue();
                assert yawDataQ.peek() != null;
                double yaw = yawDataQ.peek().doubleValue();
                assert rollDataQ.peek() != null;
                double roll = rollDataQ.peek().doubleValue();
                assert pitchDataQ.peek() != null;
                double pitch = pitchDataQ.peek().doubleValue();


                Sensor.SensorData latest = Sensor.SensorData.createWithDefaults(accX, accY, accZ, gyroX, gyroY, gyroZ, yaw, roll, pitch);
                updateSensorTextFields(latest);
            } else {
                Sensor.SensorData latest = Sensor.SensorData.createWithDefaults(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                updateSensorTextFields(latest);
            }
        }
    }


    /**
     * Updates location text fields with the current location's coordinates.
     */
    private void updateLocationTextFields() {
        if (latitudeText != null) {
            latitudeText.setText(String.format("Latitude: %.4f", location.getLatitude()));
        }
        if (longitudeText != null) {
            longitudeText.setText(String.format("Longitude: %.4f", location.getLongitude()));
        }
    }


    /**
     * Creates a toggle button with the specified text and selection state.
     *
     * @param text     the button text
     * @param selected the initial selection state
     * @return the configured {@link ToggleButton}
     */
    private ToggleButton createToggleButton(String text, boolean selected) {
        ToggleButton toggle = new ToggleButton(text);
        toggle.setSelected(selected);
        return toggle;
    }


    /**
     * Adds sensor data to graph series for visualization.
     */
    private void addDataToSeries() {
        if (!accXDataQ.isEmpty()) {
            if (isMPUOn) {
                accXSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), accXDataQ.remove().doubleValue() + 5));
                accYSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), accYDataQ.remove()));
                accZSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), accZDataQ.remove().doubleValue() - 5));
                gyroXSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), gyroXDataQ.remove().doubleValue() + 300));
                gyroYSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), gyroYDataQ.remove()));
                gyroZSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), gyroZDataQ.remove().doubleValue() - 300));
                pitchSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), pitchDataQ.remove()));
                yawSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), yawDataQ.remove()));
                rollSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), rollDataQ.remove()));
            } else {
                accXDataQ.remove();
                accYDataQ.remove();
                accZDataQ.remove();
                gyroXDataQ.remove();
                gyroYDataQ.remove();
                gyroZDataQ.remove();
                pitchDataQ.remove();
                yawDataQ.remove();
                rollDataQ.remove();
            }
            // Add altitude data regardless of MPU state, as it's from BMP sensor
            altitudeSeries.getData().add(new XYChart.Data<>(String.valueOf(xSeriesData), altitudeDataQ.remove()));
            xSeriesData++;
        }


        if (accXSeries.getData().size() > MAX_DATA_POINTS) {
            accXSeries.getData().remove(0, accXSeries.getData().size() - MAX_DATA_POINTS);
            accYSeries.getData().remove(0, accYSeries.getData().size() - MAX_DATA_POINTS);
            accZSeries.getData().remove(0, accZSeries.getData().size() - MAX_DATA_POINTS);
            gyroXSeries.getData().remove(0, gyroXSeries.getData().size() - MAX_DATA_POINTS);
            gyroYSeries.getData().remove(0, gyroYSeries.getData().size() - MAX_DATA_POINTS);
            gyroZSeries.getData().remove(0, gyroZSeries.getData().size() - MAX_DATA_POINTS);
            pitchSeries.getData().remove(0, pitchSeries.getData().size() - MAX_DATA_POINTS);
            rollSeries.getData().remove(0, rollSeries.getData().size() - MAX_DATA_POINTS);
            yawSeries.getData().remove(0, yawSeries.getData().size() - MAX_DATA_POINTS);
            altitudeSeries.getData().remove(0, altitudeSeries.getData().size() - MAX_DATA_POINTS);
        }
    }


    /**
     * Updates the UI based on the system's power state.
     */
    private void updatePowerUI() {
        // Enable powerOn when power is off, enable powerOff when power is on
        if (powerOn != null) powerOn.setDisable(isPowerOn);
        if (powerOff != null) powerOff.setDisable(!isPowerOn);


        // Disable other buttons when power is off
        if (mpuButton != null) mpuButton.setDisable(!isPowerOn);
        if (bmpButton != null) bmpButton.setDisable(!isPowerOn);
        if (tlmButton != null) tlmButton.setDisable(!isPowerOn);
        if (gpsButton != null) gpsButton.setDisable(!isPowerOn);
        if (recordButton != null) recordButton.setDisable(!isPowerOn);
        if (voltageText != null) {
            voltageText.setText("Voltage: " + (isPowerOn ? "12.00V" : "0.00V"));
        }

        if (!isPowerOn) {
            // Clear
            accXSeries.getData().clear();
            accYSeries.getData().clear();
            accZSeries.getData().clear();
            gyroXSeries.getData().clear();
            gyroYSeries.getData().clear();
            gyroZSeries.getData().clear();
            pitchSeries.getData().clear();
            yawSeries.getData().clear();
            rollSeries.getData().clear();
            altitudeSeries.getData().clear();

            // Clear
            accXDataQ.clear();
            accYDataQ.clear();
            accZDataQ.clear();
            gyroXDataQ.clear();
            gyroYDataQ.clear();
            gyroZDataQ.clear();
            yawDataQ.clear();
            rollDataQ.clear();
            pitchDataQ.clear();
            altitudeDataQ.clear();

            // Reset sensor text fields to "OFF"
            if (accXText != null) accXText.setText("AccX: OFF");
            if (accYText != null) accYText.setText("AccY: OFF");
            if (accZText != null) accZText.setText("AccZ: OFF");
            if (gyroXText != null) gyroXText.setText("GyroX: OFF");
            if (gyroYText != null) gyroYText.setText("GyroY: OFF");
            if (gyroZText != null) gyroZText.setText("GyroZ: OFF");
            if (yawText != null) yawText.setText("Yaw: OFF");
            if (rollText != null) rollText.setText("Roll: OFF");
            if (pitchText != null) pitchText.setText("Pitch: OFF");


        } else {
            // Reload sensor data
            List<Sensor.SensorData> sensorData = RocketDashboardApp.getSensorData();
            if (sensorData != null && !sensorData.isEmpty()) {
                updateSensorData(sensorData);
                updateSensorTextFields(sensorData.getLast());
            }
        }
    }

    /**
     * Updates internal data queues with sensor data.
     *
     * @param sensorData list of {@link Sensor.SensorData} objects
     */
    private void updateSensorData(List<Sensor.SensorData> sensorData) {
        for (Sensor.SensorData data : sensorData) {
            accXDataQ.add(data.accX());
            accYDataQ.add(data.accY());
            accZDataQ.add(data.accZ());
            gyroXDataQ.add(data.gyroX());
            gyroYDataQ.add(data.gyroY());
            gyroZDataQ.add(data.gyroZ());
            yawDataQ.add(data.yaw());
            rollDataQ.add(data.roll());
            pitchDataQ.add(data.pitch());
            altitudeDataQ.add(data.altitude());
        }
    }
}