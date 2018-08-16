package uk.ac.warwick.sip.mcmcProcessing;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.UniformDistribution;

/**CLASS: BROWNIAN MOTION
 * Simulation of brownian motion, click to instantiate a particle
 */
public class BrownianMotion extends McmcApplet{
  
  //for brownian motion, use uniform target
  static public final UniformDistribution UNIFORM_DISTRIBUTION = new UniformDistribution(2);
  static public final double PROPOSAL_VARIANCE = 1000.0;
  
  /**IMPLEMENTED: DRAW MCMC
   * Draw all samples
   */
  @Override
  protected void drawMcmc() {
    this.drawAllSamples();
  }
  
  /**OVERRIDE: MOUSE RELEASED
   * Instantiate a new brownian motion
   */
  @Override
  public void mouseReleased() {
    //if the mouse has not clicked on a gui, or on a gui, instantiate a new brownian motion
    if (!this.isMouseClickOnGui) {
      if (!this.isMouseOnGui()) {
        //get mouse coordinate
        double [] mousePosition = new double [2];
        mousePosition[0] = (double) this.mouseX;
        mousePosition[1] = (double) this.mouseY;
        
        //instantiate new brownian motion
        MersenneTwister rng = new MersenneTwister(this.millis());
        this.chain = new uk.ac.warwick.sip.mcmc.RandomWalkMetropolisHastings(UNIFORM_DISTRIBUTION,
            CHAIN_LENGTH, SimpleMatrix.identity(2).scale(PROPOSAL_VARIANCE), rng);
        this.chain.setInitialValue(mousePosition);
        this.isInit = true;
      }
    }
    super.mouseReleased();
  }
  
  
  public static void main(String[] args) {
    PApplet.main("uk.ac.warwick.sip.mcmcProcessing.BrownianMotion");
  }
  
}
