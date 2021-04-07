package uk.ac.warwick.sip.mcmcprocessing;

import java.util.Iterator;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.NormalDistribution;
import uk.ac.warwick.sip.mcmc.TargetDistribution;

/**CLASS: ELLIPTICAL SLICE SAMPLING
 * Simulation for elliptical slice sampling, click to add a chain
 * Draws ellipse search space, draw red dots for rejected samples
 */
public class EllipticalSlice extends McmcApplet {

  static public final int N_PLOT = 1000;

  //chain to simulate (hides the super class version)
  uk.ac.warwick.sip.mcmc.EllipticalSlice chain;
  //target distribution
  protected TargetDistribution target;
  //Normal prior distribution,
  protected NormalDistribution prior;
  //likelihood function
  protected TargetDistribution likelihood;

  /**OVERRIDE: SETUP
   * Set target distribution, prior and likelihood
   */
  @Override
  public void setup() {
    super.setup();
    this.target = this.getNormalDistribution();
    this.prior = this.getPrior();
    this.likelihood = this.getLikelihood();
  }

  /**OVERRIDE: DRAW MCMC
   * Draw all samples except for the last one
   * Draw leap frog steps
   * Draw accepted sample from the leap frog steps
   */
  @Override
  protected void drawMcmc() {
    double [] chainArray = this.chain.getChain();
    this.drawAllButLastSamples();

    //draw all points on the ellipse
    Iterator<SimpleMatrix> ellipticalPoints = this.chain.getEllipticalPositionsIterator();
    this.stroke(255, 0, 0);
    this.fill(255, 0, 0);
    float x, y;
    while(ellipticalPoints.hasNext()) {
      SimpleMatrix point = ellipticalPoints.next();
      x = (float) point.get(0);
      y = (float) point.get(1);
      //draw the first sample
      this.ellipse(x, y , CIRCLE_SIZE, CIRCLE_SIZE);
    }

    //draw the ellipse as a parametric plot, using angle as a parameter
    //yellow line for acceptable region of the ellipse
    //red line for rejected region of the ellipse, according to slice sampling
    double angle = -Math.PI;
    double angleDiff = 2*Math.PI/N_PLOT;
    float x0 = Float.NaN;
    float y0 = Float.NaN;
    SimpleMatrix ellipticalPoint;
    this.strokeWeight(2);
    if (this.chain.getNStep() > 0) {
      for (int i=0; i<N_PLOT; i++) {
        ellipticalPoint = this.chain.getPointOnEllipse(angle);
        x = (float) ellipticalPoint.get(0);
        y = (float) ellipticalPoint.get(1);

        if (i > 0) {

          if (this.chain.isValidPointOnEllipse(ellipticalPoint)) {
            this.stroke(255, 255, 0);
          } else {
            this.stroke(255, 0, 0);
          }

          this.line(x0, y0, x, y);
        }

        x0 = x;
        y0 = y;
        angle += angleDiff;
      }
    }
    this.strokeWeight(1);

    //draw the latest sample
    this.stroke(0,255,0);
    this.fill(0,255,0);
    x = (float) chainArray[this.chain.getNStep()*2];
    y  = (float) chainArray[this.chain.getNStep()*2 + 1];
    this.ellipse(x, y , CIRCLE_SIZE, CIRCLE_SIZE);
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
        //instantiate chain
        MersenneTwister rng = new MersenneTwister(this.millis());
        this.chain = new uk.ac.warwick.sip.mcmc.EllipticalSlice(this.target, this.likelihood,
            this.prior, CHAIN_LENGTH, rng);
        this.chain.setInitialValue(mousePosition);
        super.chain = this.chain; //save copy to the superclass
        this.isInit = true;
      }
    }
    super.mouseReleased();
  }

  /**METHOD: GET LIKELIHOOD
   * Return the default likelihood function
   */
  private NormalDistribution getLikelihood() {
    //instantiate the covariance
    SimpleMatrix targetCovariance = new SimpleMatrix(2, 2);
    targetCovariance.set(0, 0, 2*NORMAL_TARGET_VARIANCE);
    targetCovariance.set(1, 1, 2*NORMAL_TARGET_VARIANCE);
    //instantiate the mean
    SimpleMatrix mean = new SimpleMatrix(2, 1, true, this.getCentre());
    return new NormalDistribution(2, mean, targetCovariance);
  }

  /**METHOD: GET PRIOR
   * Return the default prior distribution
   */
  private NormalDistribution getPrior() {
    return this.getLikelihood();
  }

  public static void main(String[] args) {
    PApplet.main("uk.ac.warwick.sip.mcmcprocessing.EllipticalSlice");
  }
}
