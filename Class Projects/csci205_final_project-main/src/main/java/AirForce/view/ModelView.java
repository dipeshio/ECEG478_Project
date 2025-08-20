/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: SooAh Lay
 * Date: 4/16/2025
 * Time: 3:27 PM
 *
 * Project: csci205_final_project
 * Package: AirForce.view
 * Class: ModelView
 *
 * Description:
 *
 * ****************************************
 */

package AirForce.view;
import AirForce.model.Sensor;

import java.util.List;

public interface ModelView {

    void visualize(List<Sensor.SensorData> data);
}