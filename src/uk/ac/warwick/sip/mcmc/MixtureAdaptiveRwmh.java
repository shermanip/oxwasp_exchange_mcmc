package uk.ac.warwick.sip.mcmc;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

/**CLASS: MIXTURE ADAPTIVE RANDOM WALK METROPOLIS HASTINGS
 * See superclass AdaptiveRwmh
 * Adaptives the proposal covariance using a mixture of
 * homogeneous rwmh and scaled chain covariance optimal for normal target
 * @see AdaptiveRwmh.java
 */
public class MixtureAdaptiveRwmh extends AdaptiveRwmh{
  
  protected double probabilitySaftey = 0.05; //probability of using homogeneous proposal step
  
  /**CONSTRUCTOR
   * See superclass AdaptiveRwmh
   * @param target See superclass RandomWalkMetropolisHastings
   * @param chainLength See superclass RandomWalkMetropolisHastings
   * @param proposalCovariance proposal covariance use in homogeneous steps
   * @param rng See superclass RandomWalkMetropolisHastings
   */
  public MixtureAdaptiveRwmh(TargetDistribution target, int chainLength, SimpleMatrix proposalCovariance,
      MersenneTwister rng){
    super(target, chainLength, proposalCovariance, rng);
  }
  
  /**OVERRIDE: ADAPTIVE STEP
   * Do a Metropolis-Hastings step but with adaptive proposal covariance
   * this.probabilitySaftey chance the proposal covarinace is the homogeneous one
   * Otherwise the proposal covariance is a scaled chain sample covariance
   */
  @Override
  public void adaptiveStep() {
    
    //declare pointer for the proposalCovarinace cholesky decomposed
    SimpleMatrix proposalCovarianceChol;
    
    //with this.probabilitySaftey chance, use the homogenous proposal covariance
    if (this.rng.nextDouble()< this.probabilitySaftey) {
      proposalCovarianceChol = this.proposalCovarianceChol;
    } else {
      //get the chain covariance and scale it so that it is optimial for targetting Normal
      SimpleMatrix proposalCovariance = new SimpleMatrix(this.chainCovariance);
      CommonOps_DDRM.scale(Math.pow(2.38, 2)/this.getNDim(), proposalCovariance.getDDRM());
      //Global.cholesky will return a null if the decomposition is unsuccessful
      //use the default proposal if a null is caught
      proposalCovarianceChol = Global.cholesky(proposalCovariance);
      if (proposalCovarianceChol == null) {
        proposalCovarianceChol = this.proposalCovarianceChol;
      }
    }
    
    //do a Metropolis-Hastings step with this proposal covariance
    this.step(proposalCovarianceChol);
  }
  
}
