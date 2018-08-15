package uk.ac.warwick.sip.mcmcProcessing;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.Mcmc;
import uk.ac.warwick.sip.mcmc.UniformDistribution;

public class BrownianMotion extends McmcApplet{
  
  protected Mcmc particle;
  protected UniformDistribution uniformDistribution = new UniformDistribution(2);
  
  @Override
  public void setup() {
    this.proposalVariance = 1000.0;
  }
  
  @Override
  protected void drawMcmc() {
    this.stroke(0,255,0);
    this.fill(0,255,0);
    float x1, x2, y1, y2;
    double [] chainArray = this.particle.getChain();
    x1 = (float) chainArray[0];
    y1 = (float) chainArray[1];
    this.ellipse(x1, y1 , CIRCLE_SIZE, CIRCLE_SIZE);
    for (int i=1; i<=this.particle.getNStep(); i++) {
      x2 = (float) chainArray[i*2];
      y2 = (float) chainArray[i*2+1];
      this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
      this.line(x1, y1, x2, y2);
      x1 = x2;
      y1 = y2;
    }
  }
  
  @Override
  protected void takeStep() {
    this.particle.step();
  }
  
  @Override
  protected void changeProperty() {
  }
  
  @Override
  public void mouseReleased() {
    double [] mousePosition = new double [2];
    mousePosition[0] = (double) this.mouseX;
    mousePosition[1] = (double) this.mouseY;
    
    if (this.mouseButton == PApplet.LEFT) {
      MersenneTwister rng = new MersenneTwister(this.millis());
      this.particle = new uk.ac.warwick.sip.mcmc.RandomWalkMetropolisHastings(this.uniformDistribution, this.chainLength
          , SimpleMatrix.identity(2).scale(this.proposalVariance)
          , rng);
      this.particle.setInitialValue(mousePosition);
      this.isInit = true;
    }
  }
  
  
  public static void main(String[] args) {
    PApplet.main("uk.ac.warwick.sip.mcmcProcessing.BrownianMotion");
  }
  
}
