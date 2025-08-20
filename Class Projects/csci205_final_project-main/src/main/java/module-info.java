module csci205_final_project {
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.media;
    requires java.net.http;
    requires java.desktop;

    opens AirForce.controller to javafx.fxml;
    exports AirForce.view;
    exports AirForce;
    exports AirForce.model;
    exports AirForce.controller;

    opens AirForce to javafx.fxml;
    opens AirForce.model to javafx.base;
    opens AirForce.view to javafx.fxml, javafx.base;
}