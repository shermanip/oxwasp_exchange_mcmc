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
  
  protected SimpleMatrix safteyProposalCovarianceChol; //proposal covariance of the saftey step
  protected double probabilitySaftey = 0.05; //probability of using the step
  
  /**CONSTRUCTOR
   * See superclass AdaptiveRwmh
   * @param target See superclass RandomWalkMetropolisHastings
   * @param chainLength See superclass RandomWalkMetropolisHastings
   * @param proposalCovariance proposal covariance use in homogeneous steps
   * @param rng See superclass RandomWalkMetropolisHastings
   */
  public MixtureAdaptiveRwmh(TargetDistribution target, int chainLength,
      SimpleMatrix proposalCovariance, MersenneTwister rng){
    super(target, chainLength, proposalCovariance, rng);
    this.safteyProposalCovarianceChol = new SimpleMatrix(this.proposalCovarianceChol);
  }
  
  /**CONSTRUCTOR
   * Constructor for extending the length of the chain and resume running it
   * Does a shallow copy of the provided chain and extending the member variable chainArray
   * @param chain Chain to be extended
   * @param nMoreSteps Number of steps to be extended
   */
  public MixtureAdaptiveRwmh(MixtureAdaptiveRwmh chain, int nMoreSteps) {
    //call superconstructor to do a shallow copy and extend the chain
    super(chain, nMoreSteps);
  //shallow copy member variables
    this.probabilitySaftey = chain.probabilitySaftey;
    this.safteyProposalCovarianceChol = chain.safteyProposalCovarianceChol;
  }
  
  /**OVERRIDE: ADAPTIVE STEP
   * Do a Metropolis-Hastings step but with adaptive proposal covariance
   * this.probabilitySaftey chance the proposal covarinace is the homogeneous one
   * Otherwise the proposal covariance is a scaled chain sample covariance
   * @param currentStep Column vector of the current step of the MCMC, to be modified
   */
  @Override
  public void adaptiveStep(SimpleMatrix currentStep) {
    
    //with this.probabilitySaftey chance, use the saftey proposal covariance
    if (this.rng.nextDouble()< this.probabilitySaftey) {
      this.proposalCovarianceChol = this.safteyProposalCovarianceChol;
    } else {
      //get the chain covariance and scale it so that it is optimial for targetting Normal
      this.proposalCovarianceChol = new SimpleMatrix(this.chainCovariance);
      CommonOps_DDRM.scale(Math.pow(2.38, 2)/this.getNDim(), this.proposalCovarianceChol.getDDRM());
      //Global.cholesky will return a null if the decomposition is unsuccessful
      //use the default proposal if a null is caught
      this.proposalCovarianceChol = Global.cholesky(this.proposalCovarianceChol);
      if (this.proposalCovarianceChol == null) {
        this.proposalCovarianceChol = this.safteyProposalCovarianceChol;
      }
    }
    
    //do a Metropolis-Hastings step with this proposal covariance
    this.metropolisHastingsStep(currentStep);
  }
  
}
