/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: Carlos Stolper
 * Date: 5/1/25
 * Time: 1:35 PM
 *
 * Project: csci205_final_project
 * Package: AirForce.view
 * Class: AccelPlotAppTest
 *
 * Description:
 *
 * ****************************************
 */
package AirForce.view;

import AirForce.model.Sensor.SensorData;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AccelPlotAppTest {

    @Test
    void setDataShouldCopyList() throws Exception {
        // Create a SensorData record directly with all 13 components:
        SensorData sample = new SensorData(
                1,    // id
                0.1,  // accX
                0.2,  // accY
                0.3,  // accZ
                0.4,  // gyroX
                0.5,  // gyroY
                0.6,  // gyroZ
                0.7,  // yaw
                0.8,  // pitch
                0.9,  // roll
                1.0,  // pressure
                2.0,  // temperature
                3.0   // altitude
        );

        List<SensorData> original = new ArrayList<>();
        original.add(sample);

        // Execute
        AccelPlotApp.setData(original);
        // Mutate original
        original.clear();

        // Reflect into the private static field
        Field f = AccelPlotApp.class.getDeclaredField("data");
        f.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<SensorData> internal = (List<SensorData>) f.get(null);

        assertNotNull(internal, "Internal data list should not be null");
        assertEquals(1, internal.size(), "Internal list must still have the element");

        SensorData kept = internal.get(0);
        assertEquals(1,    kept.id());
        assertEquals(0.1,  kept.accX());
        assertEquals(0.5,  kept.gyroY());
        assertEquals(3.0,  kept.altitude());
    }
}
