package uk.ac.warwick.sip.mcmcProcessing;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.Mcmc;
import uk.ac.warwick.sip.mcmc.UniformDistribution;

public class BrownianMotion extends McmcApplet{
  
  protected UniformDistribution uniformDistribution = new UniformDistribution(2);
  protected double proposalVariance = 1000.0;
  
  @Override
  protected void drawMcmc() {
    this.drawAllSamples();
  }
  
  
  @Override
  protected void changeProperty() {
  }
  
  @Override
  public void mouseReleased() {
    if (!this.isMouseClickOnGui) {
      if (!this.isMouseOnGui()) {
        double [] mousePosition = new double [2];
        mousePosition[0] = (double) this.mouseX;
        mousePosition[1] = (double) this.mouseY;
        
        MersenneTwister rng = new MersenneTwister(this.millis());
        this.chain = new uk.ac.warwick.sip.mcmc.RandomWalkMetropolisHastings(this.uniformDistribution, this.chainLength
            , SimpleMatrix.identity(2).scale(this.proposalVariance)
            , rng);
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
