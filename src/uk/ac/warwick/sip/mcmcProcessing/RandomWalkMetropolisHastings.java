package uk.ac.warwick.sip.mcmcProcessing;

import org.apache.commons.math3.random.MersenneTwister;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.TargetDistribution;

public class RandomWalkMetropolisHastings extends McmcApplet{
  
  protected uk.ac.warwick.sip.mcmc.RandomWalkMetropolisHastings chain;
  protected TargetDistribution target;
  
  
  @Override
  public void setup() {
    this.target = this.getNormalDistribution();
  }
  
  @Override
  protected void drawMcmc() {
    this.stroke(0,255,0);
    this.fill(0,255,0);
    float x1, x2, y1, y2;
    double [] chainArray = this.chain.getChain();
    x1 = (float) chainArray[0];
    y1 = (float) chainArray[1];
    this.ellipse(x1, y1 , CIRCLE_SIZE, CIRCLE_SIZE);
    for (int i=1; i<=this.chain.getNStep(); i++) {
      x2 = (float) chainArray[i*2];
      y2 = (float) chainArray[i*2+1];
      this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
      this.line(x1, y1, x2, y2);
      x1 = x2;
      y1 = y2;
    }
    if (!this.chain.getIsAccepted()) {
      double [] rejectedSample = this.chain.getRejectedSample();
      x2 = (float) rejectedSample[0];
      y2 = (float) rejectedSample[1];
      this.stroke(255, 0, 0);
      this.fill(255, 0, 0);
      this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
      this.line(x1, y1, x2, y2);
    }
  }
  
  @Override
  protected void takeStep() {
    this.chain.step();
  }
  
  @Override
  protected void changeProperty() {
    this.chain.setProposalCovariance(this.getProposalCovariance());
  }
  
  @Override
  public void mouseReleased() {
    double [] mousePosition = new double [2];
    mousePosition[0] = (double) this.mouseX;
    mousePosition[1] = (double) this.mouseY;
    
    if (this.mouseButton == PApplet.LEFT) {
      MersenneTwister rng = new MersenneTwister(this.millis());
      this.chain = new uk.ac.warwick.sip.mcmc.RandomWalkMetropolisHastings(this.target, this.chainLength
          , this.getProposalCovariance(), rng);
      this.chain.setInitialValue(mousePosition);
      this.isInit = true;
    }
  }
  
  
  public static void main(String[] args) {
    PApplet.main("uk.ac.warwick.sip.mcmcProcessing.RandomWalkMetropolisHastings");
  }
  
}
