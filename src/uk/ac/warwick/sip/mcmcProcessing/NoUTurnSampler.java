package uk.ac.warwick.sip.mcmcProcessing;

import java.util.Iterator;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.TargetDistribution;

/**CLASS: NO U TURN SAMPLER
 * Simulation for NUTS, click to start a chain
 */
public class NoUTurnSampler extends McmcApplet{
  
  //chain to simulate (hides the super class version)
  uk.ac.warwick.sip.mcmc.NoUTurnSampler chain;
  
  protected TargetDistribution target;
  
  /**OVERRIDE: SETUP
   * Set target distribution
   */
  @Override
  public void setup() {
    super.setup();
    this.target = this.getNormalDistribution();
  }
  
  /**OVERRIDE: DRAW MCMC
   * Draw all samples except for the last one
   * Draw leap frog steps
   * Draw accepted sample from the leap frog steps
   */
  @Override
  protected void drawMcmc() {
    
    float x1, x2, y1, y2;
    double [] chainArray = this.chain.getChain();
    //draw all the samples except for the last one
    this.drawAllButLastSamples();
    
    //if the sample is accepted
    if (this.chain.getIsAccepted()) {
      //in yellow
      this.stroke(255,255,0);
      this.fill(255,255,0);
      //draw all leap frog positions
      double [] leapFrogPosition;
      Iterator<SimpleMatrix> leapFrogPositionIterator =
          ((uk.ac.warwick.sip.mcmc.NoUTurnSampler) this.chain).getLeapFrogPositionIterator();
      //get the first leap frog
      leapFrogPosition = leapFrogPositionIterator.next().getDDRM().getData();
      x1 = (float) leapFrogPosition[0];
      y1 = (float) leapFrogPosition[1];
      //draw the first sample
      this.ellipse(x1, y1 , CIRCLE_SIZE, CIRCLE_SIZE);
      //draw line and circle for the rest of the leap frog steps
      while (leapFrogPositionIterator.hasNext()) {
        leapFrogPosition = leapFrogPositionIterator.next().getDDRM().getData();
        x2 = (float) leapFrogPosition[0];
        y2 = (float) leapFrogPosition[1];
        this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
        this.line(x1, y1, x2, y2);
        x1 = x2;
        y1 = y2;
      }
      //draw in blue the accepted sample
      this.stroke(0,0,255);
      this.fill(0,0,255);
      x2 = (float) chainArray[this.chain.getNStep()*2];
      y2 = (float) chainArray[this.chain.getNStep()*2+1];
      this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
    }
  }
  
  /**OVERRIDE: MOUSE RELEASED
   * Instantiate a new chain, if the mouse of not or clicked on a gui
   */
  @Override
  public void mouseReleased() {
    //if the mouse hasn't clicked on a gui or on one
    if (!this.isMouseClickOnGui) {
      if (!this.isMouseOnGui()) {
        //get mouse position
        double [] mousePosition = new double [2];
        mousePosition[0] = (double) this.mouseX;
        mousePosition[1] = (double) this.mouseY;
        //instantiate nuts
        MersenneTwister rng = new MersenneTwister(this.millis());
        this.chain = new uk.ac.warwick.sip.mcmc.NoUTurnSampler(this.target
            , CHAIN_LENGTH, SimpleMatrix.identity(2).scale(MASS_HMC), SIZE_LEAP_FROG
            , rng);
        this.chain.setInitialValue(mousePosition);
        super.chain = this.chain; //save copy to the superclass
        this.isInit = true;
      }
    }
    super.mouseReleased();
  }
  
  
  public static void main(String[] args) {
    PApplet.main("uk.ac.warwick.sip.mcmcProcessing.NoUTurnSampler");
  }
  
}