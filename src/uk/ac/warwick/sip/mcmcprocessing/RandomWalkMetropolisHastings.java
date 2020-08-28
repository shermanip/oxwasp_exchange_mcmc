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

import g4p_controls.GCheckbox;
import g4p_controls.GEvent;
import g4p_controls.GToggleControl;
import processing.core.PApplet;
import processing.core.PVector;
import uk.ac.warwick.sip.mcmc.TargetDistribution;

/**CLASS: RANDOM WALK METROPOLIS HASTINGS
 * Simulation of rwmh, with adaptive option
 * Click and drag to instantiate a chain, longer drag for bigger proposal
 */
public class RandomWalkMetropolisHastings extends McmcApplet{

  //chain to run (this hides the super class version)
  uk.ac.warwick.sip.mcmc.MixtureAdaptiveRwmh chain;

  //target distribution is normal
  protected TargetDistribution target;
  //indicate if the mouse is being dragged and setting the proposal variance
  protected boolean isCreatingChain = false;
  //position of the mouse when it is clicked
  protected double [] clickPosition = new double[2];

//gui for toggling adapting
  protected GCheckbox adaptiveCheckBox;

  /**OVERRIDE: SETUP
   * Set the target and the GUI for toggle adaptive option
   */
  @Override
  public void setup() {
    super.setup();
    this.target = this.getNormalDistribution();
    this.adaptiveCheckBox = new GCheckbox(this, 12,150,75,40,"Adaptive"); //set position and size
    this.adaptiveCheckBox.setLocalColorScheme(255); //set colour
  }

  /**IMPLEMENTED: DRAW MCMC
   * Draw all samples in green, draw rejected samples in red
   */
  @Override
  protected void drawMcmc() {
    //if the mouse isn't being dragged to set the proposal variance
    if (!this.isCreatingChain) {
      //draw all samples
      double [] lastSample = this.drawAllSamples();
      //if a sample is rejected
      if (!this.chain.getIsAccepted()) {
        //draw rejected sample in red
        double [] rejectedSample = this.chain.getRejectedSample();
        float x1 = (float) lastSample[0];
        float y1 = (float) lastSample[1];
        float x2 = (float) rejectedSample[0];
        float y2 = (float) rejectedSample[1];
        this.stroke(255, 0, 0);
        this.fill(255, 0, 0);
        this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
        this.line(x1, y1, x2, y2);
      }
    }
  }

  /**OVERRIDE: DRAW OTHER GUI
   * Draw a circle, centred where the mouse is pressed, mouse at the moment touches the circle
   * This indicates the size of the proposal
   */
  @Override
  protected void drawOtherGui() {
    //if the mouse is being dragged, draw the circle
    if (this.isCreatingChain) {
      this.stroke(0, 255, 0);
      float radius = this.getMouseDragDistance();
      this.ellipse((float)this.clickPosition[0], (float)this.clickPosition[1], 2*radius, 2*radius);
    }
  }

  /**METHOD: GET MOUSE DRAG DISTANCE
   * Calculate the distance between where the mouse is pressed, to the current mouse position
   * @return Distance from where the mouse is pressed, to the current mouse position
   */
  public float getMouseDragDistance() {
    //get vector pointing from click to mouse
    PVector r = new PVector();
    r.set((float)this.clickPosition[0]-this.mouseX, (float)this.clickPosition[1]-this.mouseY);
    //if the distance is less than a pixel, set that distance to be one
    if (r.mag()<1) {
      return (float) 0.5;
    } else {
      return r.mag();
    }
  }

  /**OVERRIDE: IS MOUSE ON GUI
   * Update the method to indicate if the mouse is over the check box
   */
  @Override
  protected boolean isMouseOnGui() {
    boolean isMouseOnGui = super.isMouseOnGui();
    if (this.adaptiveCheckBox.isOver(this.mouseX, this.mouseY)) {
      isMouseOnGui = true;
    }
    return isMouseOnGui;
  }

  /**OVERRIDE: MOUSE PRESSED
   * If the mouse isn't pressed on a gui, record where it is click and begin to create a chain
   * The proposal variance depends on the mouse drag distance
   */
  @Override
  public void mousePressed() {
    super.mousePressed();
    if (!this.isMouseClickOnGui) {
      this.isCreatingChain = true;
      this.clickPosition[0] = (double) this.mouseX;
      this.clickPosition[1] = (double) this.mouseY;
    }
  }

  /**OVERRIDE: MOUSE RELEASED
   * If the mouse hasn't been clicked on a gui, instantiate a chain, mouse drag distance correspond
   * to the proposal std
   */
  @Override
  public void mouseReleased() {
    //if the mouse isn't clicked on a gui
    if (!this.isMouseClickOnGui) {
      //instantiate rwmh
      MersenneTwister rng = new MersenneTwister(this.millis());
      //proposal std is the distance the mouse is dragged
      double proposalVariance = Math.pow((double) this.getMouseDragDistance(),2);
      SimpleMatrix proposalCovariance = new SimpleMatrix(2,2);
      proposalCovariance.set(0,proposalVariance);
      proposalCovariance.set(3,proposalVariance);
      this.chain = new uk.ac.warwick.sip.mcmc.MixtureAdaptiveRwmh(this.target, CHAIN_LENGTH
          , proposalCovariance, rng);
      //set the initial value and adaptive option
      this.chain.setInitialValue(this.clickPosition);
      this.chain.setIsAdaptive(this.adaptiveCheckBox.isSelected());
      //save a cast chain for the super class
      super.chain = this.chain;
      this.isInit = true;
    }
    //the chain has been created, adjust all booleans accordingly
    this.isCreatingChain = false;
    super.mouseReleased();
  }

  /**METHOD: HANDLE TOGGLE CONTROL EVENTS
   * See G4P library, called when a checkbox has been interacted
   * @param checkbox The checkbox being interacted
   * @param event What has happened to the checkbox
   */
  public void handleToggleControlEvents(GToggleControl checkbox, GEvent event) {
    //if the checkbox being interacted is the adaptive check box
    if (checkbox == this.adaptiveCheckBox) {
      //if a chain has been instantiate, set the chain to be (or not to be) adaptive
      if (this.isInit) {
        this.chain.setIsAdaptive(this.adaptiveCheckBox.isSelected());
      }
    }
  }


  public static void main() {
    PApplet.main("uk.ac.warwick.sip.mcmcprocessing.RandomWalkMetropolisHastings");
  }

}
