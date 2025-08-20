/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: Carlos Stolper
 * Date: 4/30/25
 * Time: 2:42 PM
 *
 * Project: csci205_final_project
 * Package: PACKAGE_NAME
 * Class: controller.GUIControllerTest
 *
 * Description:
 *
 * ****************************************
 */
/* *****************************************
 * CSCI 205 - Software Engineering and Design
 * Spring 2025
 *
 * Name: Carlos Stolper
 * Date: 4/30/25
 * Time: 2:42 PM
 *
 * Project: csci205_final_project
 * Package: controller
 * Class: GUIControllerTest
 *
 * Description: Tests for GUIController toggle handlers
 * ****************************************
 */

package AirForce.controller;

import AirForce.model.CountDownCheck;
import AirForce.model.SystemOperation;
import javafx.scene.control.ToggleButton;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GUIControllerTest {

    private SystemOperation sysOp;
    private CountDownCheck cdc;
    private GUIController gui;

    @BeforeEach
    void setUp() {
        sysOp = new SystemOperation();
        cdc   = new CountDownCheck(5000);
        gui   = new GUIController(sysOp, cdc);
    }

//    @Test
//    void testHandlePowerToggle() {
//        ToggleButton btn = new ToggleButton();
//        btn.setSelected(true);
//        gui.handlePowerToggle(btn);
//        assertTrue(sysOp.isPowerOn());
//        assertEquals("Power: ON", btn.getText());
//
//        btn.setSelected(false);
//        gui.handlePowerToggle(btn);
//        assertFalse(sysOp.isPowerOn());
//        assertEquals("Power: OFF", btn.getText());
//    }
//
//    @Test
//    void testHandleMPUToggle() {
//        ToggleButton btn = new ToggleButton();
//        btn.setSelected(true);
//        gui.handleMPUToggle(btn);
//        assertTrue(sysOp.isMPUEnabled());
//        assertEquals("MPU: ON", btn.getText());
//
//        btn.setSelected(false);
//        gui.handleMPUToggle(btn);
//        assertFalse(sysOp.isMPUEnabled());
//        assertEquals("MPU: OFF", btn.getText());
//    }
//
//    @Test
//    void testHandleTLMToggle() {
//        ToggleButton btn = new ToggleButton();
//        btn.setSelected(true);
//        gui.handleTLMToggle(btn);
//        assertTrue(sysOp.isTLMEnabled());
//        assertEquals("TLM: 500", btn.getText());
//
//        btn.setSelected(false);
//        gui.handleTLMToggle(btn);
//        assertFalse(sysOp.isTLMEnabled());
//        assertEquals("TLM: 20", btn.getText());
//    }
//
//    @Test
//    void testHandleGPSToggle() {
//        ToggleButton btn = new ToggleButton();
//        btn.setSelected(true);
//        gui.handleGPSToggle(btn);
//        assertTrue(sysOp.isGPSEnabled());
//        assertEquals("GPS: ON", btn.getText());
//
//        btn.setSelected(false);
//        gui.handleGPSToggle(btn);
//        assertFalse(sysOp.isGPSEnabled());
//        assertEquals("GPS: OFF", btn.getText());
//    }
//
//    @Test
//    void testHandleBMPToggle() {
//        ToggleButton btn = new ToggleButton();
//        btn.setSelected(true);
//        gui.handleBMPToggle(btn);
//        assertTrue(sysOp.isBMPEnabled());
//        assertEquals("BMP: ON", btn.getText());
//
//        btn.setSelected(false);
//        gui.handleBMPToggle(btn);
//        assertFalse(sysOp.isBMPEnabled());
//        assertEquals("BMP: OFF", btn.getText());
//    }
}