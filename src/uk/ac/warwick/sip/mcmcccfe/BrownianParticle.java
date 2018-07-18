package uk.ac.warwick.sip.mcmcccfe;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

public class BrownianParticle extends HomogeneousRwmh{
  
  protected SimpleMatrix attractionVector;
  protected boolean hasDrift = false;
  protected double driftMagnitude;
  
  public BrownianParticle(int nDim, int chainLength, SimpleMatrix proposalCovariance,
      double driftMagnitude, MersenneTwister rng) {
    super(new UniformDistribution(nDim), chainLength, proposalCovariance, rng);
    this.driftMagnitude = driftMagnitude;
  }
  
  public void setAttractionVector(double [] attractionVector) {
    this.attractionVector = new SimpleMatrix(this.getNDim(), 1, true, attractionVector);
  }
  
  public void setHasDrift(boolean hasDrift) {
    this.hasDrift = hasDrift;
  }
  
  public void setDriftMagnitude(double driftMagnitude) {
    this.driftMagnitude = driftMagnitude;
  }
  
  public boolean getHasDrift() {
    return this.hasDrift;
  }
  
  public double [] getAttractionVector() {
    return this.attractionVector.getDDRM().getData();
  }
  
  /**OVERRIDE: STEP
   */
  @Override
  public void step(){
    super.step();
    //if there are MCMC steps to do...
    if (this.nStep+1 < this.chainLength) {
      if (this.hasDrift) {
        SimpleMatrix chainPosition = this.chainArray.extractVector(true, this.nStep);
        CommonOps_DDRM.transpose(chainPosition.getDDRM());
        SimpleMatrix drift = this.attractionVector.minus(chainPosition);
        SimpleMatrix driftSquared = new SimpleMatrix(this.getNDim(),1);
        CommonOps_DDRM.elementPower(drift.getDDRM(), 2, driftSquared.getDDRM());
        double abs = Math.sqrt(CommonOps_DDRM.elementSum(driftSquared.getDDRM()));
        CommonOps_DDRM.scale(this.driftMagnitude/abs, drift.getDDRM());
        for (int i=0; i<this.getNDim(); i++) {
          this.chainArray.set(this.nStep, i, this.chainArray.get(this.nStep, i)+drift.get(i));
        }
      }
    }
  }
  
}
