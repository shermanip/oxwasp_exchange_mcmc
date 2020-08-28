/*
 *    Copyright 2018-2020 Sherman Lo

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

package uk.ac.warwick.sip.mcmcprocessing;

import java.util.HashMap;

import g4p_controls.GButton;
import g4p_controls.GEvent;
import processing.core.PApplet;

public class Menu extends PApplet {

  static private String[] KEYS = {"-bm", "-rwmh", "-hmc", "-nuts"};
  static public final int [] BACKGROUND_COLOUR = {0,33,71};

  //buttons for the difference mcmc
  private HashMap<String, String> mcmcNameMap;
  private HashMap<String, GButton> buttonMap;
  private String[] args;
  private boolean isMadeSelection = false;


  /**OVERRIDE: SETTINGS
   * Set the size of the applet
   */
  @Override
  public void settings() {
    this.size(700, 430);
  }

  /**OVERRIDE: SETUP
   * Instantiate the GUI
   */
  @Override
  public void setup() {
    String[] names = {
      "Brownian Motion",
      "Random Walk Metrpolis-Hastings",
      "Hamiltonian Monte Carlo",
      "No U-Turn Sampler"
    };

    this.mcmcNameMap = new HashMap<String, String>();
    for (int i=0; i<KEYS.length; i++) {
      this.mcmcNameMap.put(KEYS[i], names[i]);
    }

    this.buttonMap = new HashMap<String, GButton>();
    int yStart = 120;
    int height = 60;
    int spacing = 20;
    for (int i=0; i<KEYS.length; i++) {
      GButton button = new GButton(this, 10, yStart+(height+spacing)*i, this.width-20, height,
          this.mcmcNameMap.get(KEYS[i]));
      this.buttonMap.put(KEYS[i], button);
    }
  }

  /**OVERRIDE: DRAW
   * Draw at every frame
   */
  @Override
  public void draw() {
    this.background(BACKGROUND_COLOUR[0], BACKGROUND_COLOUR[1], BACKGROUND_COLOUR[2]);
    this.textSize(32);
    this.text("Markov Chains using Monte Carlo in Java", 10, 40);
    this.textSize(16);
    this.text("Copyright 2018-2020 Sherman Lo", 10, 84);
  }

  /**METHOD: HANDLE BUTTON EVENTS
   * See G4P library, called when a button has been interacted
   * @param button Interacted button
   * @param event Variable indicating what happened
   */
  public void handleButtonEvents(GButton button, GEvent event) {
    //when a button has been clicked
    if (event == GEvent.CLICKED) {
      for (int i=0; i<KEYS.length; i++) {
        if (button == buttonMap.get(KEYS[i])) {
          this.args = new String[1];
          this.args[0] = KEYS[i];
          this.isMadeSelection = true;
        }
      }
    }
  }

  public boolean isMadeSelection() {
    return this.isMadeSelection;
  }

  public String[] getArgs() {
    return this.args;
  }

  public static void main(String[] args) {
    Menu menu = new uk.ac.warwick.sip.mcmcprocessing.Menu();
    String[] name = {"uk.ac.warwick.sip.mcmcprocessing.Menu"};
    PApplet.runSketch(name, menu);
    while (true) {
      if (menu.isMadeSelection()) {
        args = menu.getArgs();
        menu.exit();
        uk.ac.warwick.sip.mcmc.Global.main(args);
        break;
      }
    }
  }

}
