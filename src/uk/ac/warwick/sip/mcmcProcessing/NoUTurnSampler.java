package uk.ac.warwick.sip.mcmcProcessing;

import java.util.Iterator;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.TargetDistribution;

public class NoUTurnSampler extends McmcApplet{
  
  protected TargetDistribution target;
  protected double targetVariance = 1000;
  
  
  @Override
  public void setup() {
    super.setup();
    this.target = this.getNormalDistribution();
  }
  
  @Override
  protected void drawMcmc() {
    float x1, x2, y1, y2;
    double [] chainArray = this.chain.getChain();
    this.drawAllButLastSamples();
    if (this.chain.getIsAccepted()) {
      
      this.stroke(255,255,0);
      this.fill(255,255,0);
      double [] leapFrogPosition;
      Iterator<SimpleMatrix> leapFrogPositionIterator =
          ((uk.ac.warwick.sip.mcmc.NoUTurnSampler) this.chain).getLeapFrogPositionIterator();
      leapFrogPosition = leapFrogPositionIterator.next().getDDRM().getData();
      x1 = (float) leapFrogPosition[0];
      y1 = (float) leapFrogPosition[1];
      this.ellipse(x1, y1 , CIRCLE_SIZE, CIRCLE_SIZE);
      while (leapFrogPositionIterator.hasNext()) {
        leapFrogPosition = leapFrogPositionIterator.next().getDDRM().getData();
        x2 = (float) leapFrogPosition[0];
        y2 = (float) leapFrogPosition[1];
        this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
        this.line(x1, y1, x2, y2);
        x1 = x2;
        y1 = y2;
      }
      
      this.stroke(0,0,255);
      this.fill(0,0,255);
      x2 = (float) chainArray[this.chain.getNStep()*2];
      y2 = (float) chainArray[this.chain.getNStep()*2+1];
      this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
    }
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
        this.chain = new uk.ac.warwick.sip.mcmc.NoUTurnSampler(this.target
            , this.chainLength, this.getProposalCovariance(), SIZE_LEAP_FROG
            , rng);
        this.chain.setInitialValue(mousePosition);
        this.isInit = true;
      }
    }
    super.mouseReleased();
  }
  
  
  public static void main(String[] args) {
    PApplet.main("uk.ac.warwick.sip.mcmcProcessing.NoUTurnSampler");
  }
  
}