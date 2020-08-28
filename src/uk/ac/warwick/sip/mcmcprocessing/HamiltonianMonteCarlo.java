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

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

import g4p_controls.G4P;
import g4p_controls.GEvent;
import g4p_controls.GSlider;
import g4p_controls.GValueControl;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.TargetDistribution;

/**CLASS: HAMILTONIAN MONTE CARLO
 * Simulation for HMC, click to add a chain
 * Slider bar can adjust the number of leap frog steps per HMC step
 */
public class HamiltonianMonteCarlo extends McmcApplet{

  //chain to simulate (hides the super class version)
  uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo chain;

  //target distribution
  protected TargetDistribution target;
  //keep tracks of the number of leap frog step
  //values shouldn't be extracted the slider as it will change during pause
  protected int nLeapFrog = 10;

  //gui to adjust the number of leap frog steps
  protected GSlider nLeapFrogSlider;

  /**OVERRIDE: SETUP
   * Set the GUI for the slider
   */
  @Override
  public void setup() {
    super.setup();
    target = this.getNormalDistribution();

    //instantiate slider bar
    this.nLeapFrogSlider = new GSlider(this, 110, 150, 300, 150, 30);
    this.nLeapFrogSlider.setRotation(HALF_PI);
    this.nLeapFrogSlider.setShowValue(true);
    this.nLeapFrogSlider.setLimits(10, 1,
        uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo.MAX_N_LEAP_FROG);
    this.nLeapFrogSlider.setTextOrientation(-1); //rotate text
    this.nLeapFrogSlider.setNbrTicks(11);
    this.nLeapFrogSlider.setLocalColor(2,-1); //set colour of the text
    this.nLeapFrogSlider.setShowTicks(true);
    this.nLeapFrogSlider.setNumberFormat(G4P.INTEGER, 0);
  }

  /**IMPLEMENTED: DRAW MCMC
   * Draw all samples except for the last one
   * Draw all the leap frog steps to the last sample
   */
  @Override
  protected void drawMcmc() {
    //draw all but the last samples
    double [] secondLastSample = this.drawAllButLastSamples();
    float x1, y1, x2, y2;
    x1 = (float) secondLastSample[0];
    y1 = (float) secondLastSample[1];

    //for an accept step, draw all the leap frog steps
    if (this.chain.getIsAccepted()) {
      //yellow
      this.stroke(255,255,0);
      this.fill(255,255,0);
      double [] leapFrogPosition;
      //draw all leap frog steps
      for (int i=0; i<this.nLeapFrog; i++) {
        leapFrogPosition = this.chain.getLeapFrogPositions(i);
        x2 = (float) leapFrogPosition[0];
        y2 = (float) leapFrogPosition[1];
        this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
        this.line(x1, y1, x2, y2);
        x1 = x2;
        y1 = y2;
      }
    }

    //if a step is to be taken, update nLeapFrog with the slider
    //this is required as the slider can move when paused
    if (this.isToTakeStep) {
      this.nLeapFrog = this.nLeapFrogSlider.getValueI();
    }
  }

  /**OVERRIDE: IS MOUSE ON GUI
   * Update the method to indicate if the mouse is over a GUI
   */
  @Override
  protected boolean isMouseOnGui() {
    boolean isMouseOnGui = super.isMouseOnGui();
    if (this.nLeapFrogSlider.isOver(this.mouseX, this.mouseY)) {
      isMouseOnGui = true;
    }
    return isMouseOnGui;
  }

  /**OVERRIDE: MOUSE RELEASED
   * Instantiate HMC chain, if the mouse hasn't clicked on a gui, or on a gui, instantiate HMC
   */
  @Override
  public void mouseReleased() {

    //if the mouse hasn't clicked on a gui, or on a gui, instantiate HMC
    if (!this.isMouseClickOnGui) {
      if (!this.isMouseOnGui()) {

        //get mouse coordinate
        double [] mousePosition = new double [2];
        mousePosition[0] = (double) this.mouseX;
        mousePosition[1] = (double) this.mouseY;

        //instantiate HMC
        MersenneTwister rng = new MersenneTwister(this.millis());
        this.chain = new uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo(this.target
            , CHAIN_LENGTH, SimpleMatrix.identity(2).scale(MASS_HMC), SIZE_LEAP_FROG
            , this.nLeapFrogSlider.getValueI(), rng);
        this.chain.setInitialValue(mousePosition);
        //save a copy of the pointer to the superclass
        super.chain = this.chain;
        this.isInit = true;
      }
    }
    super.mouseReleased();
  }

  /**METHOD: HANDLE SLIDER EVENTS
   * See G4P library, called when a slider bar has been interacted
   * @param slider The slider which has been interacted
   * @param event What has happened to the slider
   */
  public void handleSliderEvents(GValueControl slider, GEvent event) {
    //for the n leap frog slider
    if (slider == this.nLeapFrogSlider) {
      if (event == GEvent.VALUE_STEADY) {
        //if a chain has been instantiated
        if (this.isInit) {
          //change the number of leap frog steps
          this.chain.setNLeapFrog(this.nLeapFrogSlider.getValueI());
        }
      }
    }
  }

  public static void main(String[] args) {
    PApplet.main("uk.ac.warwick.sip.mcmcprocessing.HamiltonianMonteCarlo");
  }

}
