package uk.ac.warwick.sip.mcmcProcessing;

import org.ejml.simple.SimpleMatrix;

import g4p_controls.GButton;
import g4p_controls.GEvent;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.TargetDistribution;
import uk.ac.warwick.sip.mcmc.NormalDistribution;

/**ABSTRACT: MCMC APPLET
 * Template for MCMC simulation on a Processing applet
 */
public abstract class McmcApplet extends PApplet{
  
  static public final double NORMAL_TARGET_VARIANCE = 40000.0; //variance of the Gaussian target
  static public final double PROPOSAL_VARIANCE = 113288.0; //optimal variance
  static public final double MASS_HMC = 40000.0; //diagonal element for the mass matrix for HMC
  static public final double SIZE_LEAP_FROG = 5000.0; //size of the leap frog step for HMC and NUTS
  static public final int CIRCLE_SIZE = 10; //ellipse size of the samples
  static public final int CHAIN_LENGTH = 1024; //length of the MCMC chain
  static public final int N_CONTOUR = 10; //number of contour plots for the target distribution
  //background colours for unpaused and paused
  static public final int [] UNPAUSED_COLOUR = {0,0,0};
  static public final int [] PAUSED_COLOUR = {0,33,71};
  
  protected uk.ac.warwick.sip.mcmc.Mcmc chain; //the chain to simulate
  
  protected boolean isPaused = false; //indicate if the program is paused
  protected boolean isToTakeStep = true; //indicate if the MCMC is to take a step
  
  protected boolean isInit = false; //indicate if a chain has been instantiated
  protected boolean isDrawNormalContour = false; //indicate to draw the contour for the target
  
  //GUI buttons
  protected GButton pauseButton;
  protected GButton stepButton;
  protected GButton quitButton;
  
  //indicate if the mouse clicked on a button or another GUI
  protected boolean isMouseClickOnGui = false;
  
  /**OVERRIDE: SETTINGS
   * Set the size of the applet
   */
  @Override
  public void settings() {
    //this.fullScreen();
    this.size(1024, 768);
  }
  
  /**OVERRIDE: SETUP
   * Instantiate the GUI  
   */
  @Override
  public void setup() {
    this.pauseButton = new GButton(this, 10,10,50,50,"pause");
    this.stepButton = new GButton(this, 10,80,50,50,"step");
    this.quitButton = new GButton(this, 10, this.height-80, 50, 50, "quit");
  }
  
  /**OVERRIDE: DRAW
   * Draw at every frame
   */
  @Override
  public void draw() {
    //draw background
    if (this.isPaused) {
      this.background(PAUSED_COLOUR[0], PAUSED_COLOUR[1], PAUSED_COLOUR[2]);
    } else {
      this.background(UNPAUSED_COLOUR[0], UNPAUSED_COLOUR[1], UNPAUSED_COLOUR[2]);
    }
    //draw the target
    if (this.isDrawNormalContour) {
      this.drawNormalContour();
    }
    //draw GUI other than the member variables
    this.drawOtherGui();
    
    //if a chain has been instantiate, draw it
    if (this.isInit) {
      this.drawMcmc();
      //take a mcmc step
      if (this.isToTakeStep) {
        this.chain.step();
      }
      //if this is paused, do not take any more steps
      //this is required as the step button will change this.isToTakeStep to true
      if (this.isPaused) {
        this.isToTakeStep = false;
      }
    }
  }
  
  /**METHOD: DRAW OTHER GUI
   * To be implemented, draws right after background and before the MCMC
   */
  protected void drawOtherGui(){
    //does nothing
  }
  
  /**ABSTRACT: DRAW MCMC
   * Draw the MCMC chain, this is drawn after the background and drawOtherGUI
   */
  protected abstract void drawMcmc();
  
  /**METHOD: DRAW ALL SAMPLES
   * Draws all the MCMC samples
   * @return 2 vector, last sample
   */
  public double [] drawAllSamples() {
    //green
    this.stroke(0,255,0);
    this.fill(0,255,0);
    float x1, x2, y1, y2; //coordinates of the samples
    double [] chainArray = this.chain.getChain(); //get array of the chain
    //get the first sample and draw it
    x1 = (float) chainArray[0];
    y1 = (float) chainArray[1];
    this.ellipse(x1, y1 , CIRCLE_SIZE, CIRCLE_SIZE);
    //for each sample, draw line between samples and draw the next sample
    for (int i=1; i<=this.chain.getNStep(); i++) {
      x2 = (float) chainArray[i*2];
      y2 = (float) chainArray[i*2+1];
      this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
      this.line(x1, y1, x2, y2);
      x1 = x2;
      y1 = y2;
    }
    //save and return the last sample
    double [] lastSample = new double[2];
    lastSample[0] = x1;
    lastSample[1] = y1;
    return lastSample;
  }
  
  /**METHOD: DRAW ALL BUT LAST SAMPLES
   * Draws all the MCMC samples except for the last one
   * @return 2 vector, 2nd last sample
   */
  public double [] drawAllButLastSamples() {
    //green
    this.stroke(0,255,0);
    this.fill(0,255,0);
    float x1, x2, y1, y2; //coordinates of the samples
    double [] chainArray = this.chain.getChain(); //get array of the chain
    //get the first sample and draw it
    x1 = (float) chainArray[0];
    y1 = (float) chainArray[1];
    this.ellipse(x1, y1 , CIRCLE_SIZE, CIRCLE_SIZE);
  //for each sample, draw line between samples and draw the next sample
    for (int i=1; i<this.chain.getNStep(); i++) {
      x2 = (float) chainArray[i*2];
      y2 = (float) chainArray[i*2+1];
      this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
      this.line(x1, y1, x2, y2);
      x1 = x2;
      y1 = y2;
    }
    
    //save and return the 2nd last sample
    double [] lastSample = new double[2];
    lastSample[0] = x1;
    lastSample[1] = y1;
    return lastSample;
  }
  
  /**METHOD: GET NORMAL DISTRIBUTION
   * @return Normal distribution
   */
  protected TargetDistribution getNormalDistribution() {
    this.isDrawNormalContour = true; //indicate to draw the target
    //instantiate the covariance
    SimpleMatrix targetCovariance = new SimpleMatrix(2, 2);
    targetCovariance.set(0, 0, NORMAL_TARGET_VARIANCE);
    targetCovariance.set(1, 1, NORMAL_TARGET_VARIANCE);
    //instantiate the mean
    SimpleMatrix mean = new SimpleMatrix(2, 1, true, this.getCentre());
    return new NormalDistribution(2, mean, targetCovariance);
  }
  
  /**METHOD: GET CENTRE
   * @return 2 vector of the centre of the applet
   */
  protected double [] getCentre() {
    double [] centre = new double [2];
    centre[0] = ((double)this.width)/2;
    centre[1] = ((double)this.height)/2;
    return centre;
  }
  
  /**METHOD: DRAW NORMAL CONTOUR
   * Draw the contours of the Normal distribution
   */
  protected void drawNormalContour() {
    org.apache.commons.math3.distribution.NormalDistribution normalDistribution =
        new org.apache.commons.math3.distribution.NormalDistribution(0
            , Math.sqrt(NORMAL_TARGET_VARIANCE));
    this.stroke(255,255,255);
    this.noFill();
    double [] meanVector = this.getCentre();
    
    //cdf to plot
    double pSpacing = 0.5 /  ((double)(N_CONTOUR+1));
    double cdf = 0.5 + pSpacing;
    //for each cdf, plot ellipse
    for (int i=0; i<N_CONTOUR; i++) {
      float contourDiameter =
          (float) (normalDistribution.inverseCumulativeProbability(cdf) * 2.0);
      cdf += pSpacing;
      this.ellipse((float)meanVector[0], (float)meanVector[1], contourDiameter, contourDiameter);
    }
    
  }
  
  /**METHOD: HANDLE BUTTON EVENTS
   * See G4P library, called when a button has been interacted
   * @param button Interacted button
   * @param event Variable indicating what happened
   */
  public void handleButtonEvents(GButton button, GEvent event) {
    //when a button has been clicked
    if (event == GEvent.CLICKED) {
      //for the pause button, 
      if (button == this.pauseButton) {
        //invert isPaused and set isToTakeStep accordingly
        this.isPaused = !this.isPaused;
        this.isToTakeStep = !this.isPaused;
      //for the step button, adjust isToTakeStep
      } else if (button == this.stepButton) {
        this.isToTakeStep = true;
      //for the quit button, exit the program
      } else if (button == this.quitButton) {
        this.exit();
      }
    }
  }
  
  /**METHOD: IS MOUSE ON GUI
   * Find out if the mouse if over any of the instantiated GUI
   * Subclasses overriding this should call this in addition
   * @return true if the mouse is over a gui
   */
  protected boolean isMouseOnGui() {
    //assume the mouse if not on a gui, then test for each GUI if the mouse is over one
    boolean isMouseOnGui = false;
    if (this.pauseButton.isOver(this.mouseX, this.mouseY)) {
      isMouseOnGui = true;
    } else if (this.stepButton.isOver(this.mouseX, this.mouseY)) {
      isMouseOnGui = true;
    } else if (this.quitButton.isOver(this.mouseX, this.mouseY)) {
      isMouseOnGui = true;
    }
    return isMouseOnGui;
  }
  
  /**METHOD: MOUSE PRESSED
   * If the mouse is clicked on a gui, indicate the program so
   * Subclasses overriding this method should call this
   */
  public void mousePressed() {
    if (this.isMouseOnGui()) {
      this.isMouseClickOnGui = true;
    }
  }
  
  /**METHOD: MOUSE RELEASED
   * The mouse has been clicked, set isMouseClickOnGui to false
   * Subclasses overriding this method should call this, at the end of the method, is this because
   * overridden method may want to use isMouseClickOnGui
   */
  public void mouseReleased() {
    this.isMouseClickOnGui = false;
  }
  
}
