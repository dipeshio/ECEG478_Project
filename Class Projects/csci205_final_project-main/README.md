# Rocket Dashboard

## Team Number 06 

## Team Members and Scrum Roles
- **SooAh Lay** - *Scrum Master*  
  Sophomore Mechanical Engineering major with a Computer Science minor. Passionate about rockets, currently conducting an independent study on hybrid rocket propulsion.
- **Dipesh Bhattarai** - *Product Owner*  
  Sophomore Mathematical Economics major with a CS minor. Actively exploring personal projects with a focus on data science.
- **Carlos Stolper** - *Developer*  
  Sophomore Computer Science major. Engaged in various personal projects to enhance coding skills.
- **Jack Mclaud** - *Developer*  
  Sophomore Economics major with a CS minor. Recently completed a Data Science project and eager to contribute to team efforts.

## Project Overview
The **Rocket Dashboard** is a Java-based application developed for CSCI 205 - Software Engineering and Design at Bucknell University during the Spring 2025 semester, under the guidance of Instructor Lily Romano. This project simulates a rocket launch control interface, offering a robust, object-oriented system to monitor and manage rocket launches with real-time data visualization. Key features include sensor monitoring (pressure, temperature, accelerometer, gyro), trajectory and location tracking (latitude, longitude, velocity, altitude with maps), a countdown clock with launch status (Stand By, Ignited, Deployed), system operations (power, voltage, command modules like BMP, MPU, TLM, GPS), launch graphs (altitude vs. time), a timeline for setup milestones, recovery location tracking, and a simulated live feed from the launch site. Designed with extensibility in mind, the project provides a foundation for future GUI enhancements and potential integration with real hardware, making it a versatile tool for educational and experimental purposes.

This collaborative effort showcases the team's ability to apply software engineering principles, including modular design, version control, and agile methodologies, to create a functional and scalable application. The dashboard serves as a practical demonstration of how software can interface with simulated rocket telemetry, offering an engaging way to explore aerospace concepts through a user-friendly interface.

## Package Structure Explanation
The project is organized into a clear package structure to ensure modularity and maintainability:
- **`AirForce`**: The root package, serving as a namespace for the project, Houses UI components like MainApp (the main application class).
- **`AirForce.model`**: Contains data models such as `Sensor`, `Location`, `LiveFeed`, `SystemOperation`, `WeatherAPI`, `CountDownCheck`, `Launch Graph`, and `TimeLine`, which define the structure of rocket telemetry data and launch events.
- **`AirForce.view`**: Houses UI components like `RocketDashboardApp`, `PressurePlotApp`, and `TemperaturePlotApp`, responsible for rendering graphs and gauges.
- **`AirForce.controller`**: Includes `DashboardController` and `GUIController`, managing the logic and interaction between the UI and model data.

This structure separates concerns, making it easy to extend functionality (e.g., adding new sensors) or modify UI elements without affecting the core data logic.

## Third-Party Libraries

- **`JavaFX 17.0.2`**

    URL: https://openjfx.io/

    Includes the following modules used in the project:

  - **`javafx.controls`**: Provides UI controls like buttons, text fields, and charts.
  - **`javafx.fxml`**: Enables FXML-based UI design and loading.
  - **`javafx.media`**: Supports media playback, such as the live feed video.
  

- **`JavaFX SDK 24.0.1`**

    URL: https://openjfx.io/

    Includes the same modules (javafx.controls, javafx.fxml, javafx.media) with updated features and bug fixes for rendering UI components. Version 24.0.1 provides the latest enhancements.

- **`JUnit 5.10.2`**

    URL: https://junit.org/junit5/

    Employed for unit testing the application logic. Version 5.10.2 provides robust testing features for the project’s model and controller classes.

### External APIs Used

- OpenWeatherMap API 

  URL: https://openweathermap.org/

    Used to fetch real-time weather data such as temperature, wind speed, and visibility for a specified location, enhancing the recovery location tracking feature.


- Google Maps API 

    URL: https://developers.google.com/maps

  Utilizes the Static Maps API to display maps for location and trajectory tracking, providing visual representation of the rocket’s position.


## Running the Project
1. Clone or download the project files into your local directory:
   ```bash
   git clone https://github.com/your-repo/RocketDashboard.git
   cd RocketDashboard
   ```
2. Ensure you have Java 17+ and Gradle installed.


3. Run the application using Gradle:
   ```bash
   ./gradlew run
   ```
   - On Windows, use `gradlew.bat run` instead.
   

4. The dashboard will launch, displaying real-time simulations and interactive controls.


5. Run the test using Gradle:
```bash
./gradlew clean test
   ```
* On Windows, use `gradlew.bat clean test` instead.
* The clean task removes previous build artifacts to ensure a fresh build. 
* The test task compiles the main and test code, runs all unit tests (e.g., in src/test/java), and generates test reports in the build/reports/tests directory.

## Video Presentation
Watch our project presentation here: [\[link\]](https://mediaspace.bucknell.edu/media/CSCI+205+Team06+Video/1_dfy78uwa).