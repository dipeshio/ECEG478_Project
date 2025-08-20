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
 * Class: RocketDashboardAppTest
 *
 * Description:
 *
 * ****************************************
 */
package AirForce.view;

import AirForce.model.Sensor.SensorData;
import AirForce.model.CountDownCheck;
import AirForce.model.SystemOperation;
import AirForce.model.TimeLine;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RocketDashboardAppTest {

    private List<SensorData> sampleData;
    private TimeLine tl;
    private CountDownCheck cdc;
    private SystemOperation so;

    @BeforeEach
    void setUp() {
        // Build sample list by directly constructing the record with zeros for non-motion fields
        sampleData = new ArrayList<>();
        sampleData.add(new SensorData(
                42,   // id
                1.0,  // accX
                2.0,  // accY
                3.0,  // accZ
                4.0,  // gyroX
                5.0,  // gyroY
                6.0,  // gyroZ
                7.0,  // yaw
                8.0,  // pitch
                9.0,  // roll
                0.0,  // pressure
                0.0,  // temperature
                0.0   // altitude
        ));

        tl  = new TimeLine();
        cdc = new CountDownCheck(5000);
        so  = new SystemOperation();
    }

    @Test
    void sensorDataSetterGetter() {
        RocketDashboardApp.setSensorData(sampleData);
        // Mutate original
        sampleData.clear();

        List<SensorData> fetched = RocketDashboardApp.getSensorData();
        assertNotNull(fetched, "Getter should never return null");
        assertEquals(1, fetched.size(), "Must copy the list, not reference it");
        assertEquals(42, fetched.get(0).id());
    }

    @Test
    void timeLineSetterGetter() {
        RocketDashboardApp.setTimeLine(tl);
        assertSame(tl, RocketDashboardApp.getTimeLine());
    }

    @Test
    void countDownCheckSetterGetter() {
        RocketDashboardApp.setCountDownCheck(cdc);
        assertSame(cdc, RocketDashboardApp.getCountDownCheck());
    }

    @Test
    void systemOperationSetterGetter() {
        RocketDashboardApp.setSystemOperation(so);
        assertSame(so, RocketDashboardApp.getSystemOperation());
    }
}
