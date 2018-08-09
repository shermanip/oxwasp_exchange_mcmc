package uk.ac.warwick.sip.mcmc;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

/**CLASS: ADAPTIVE RANDOM WALK METROPOLIS HASTINGS
 * See superclass HomogeneousRwmh and RandomWalkMetropolisHastings
 * The proposal covariance is adaptive, 2*this.getNDim()-1 initial steps are homogeneous
 * Afterwards the proposal covariance is then a scale of the chain sample covariance
 */
public class AdaptiveRwmh extends RandomWalkMetropolisHastings{
  
  //initial proposal covariance decomposed  using cholesky
  protected int nStepTillAdaptive; //number of regular MH steps till use adaptive method
  private double e; //small constant to be added to the diagional of the proposal covairnace in adaptive steps
  
  
  /**CONSTRUCTOR
   * @param target See superclass RandomWalkMetropolisHastings
   * @param chainLength See superclass RandomWalkMetropolisHastings
   * @param proposalCovariance proposal_covariance
   * @param rng See superclass RandomWalkMetropolisHastings
   */
  public AdaptiveRwmh(TargetDistribution target, int chainLength, SimpleMatrix proposalCovariance,
      MersenneTwister rng ){
    //call constructors and assign member variables
    super(target, chainLength, proposalCovariance, rng);
    this.nStepTillAdaptive = 2*this.getNDim()-1; //default value
    this.e = 1E-10; //default value
  }
  
  /**CONSTRUCTOR
   * Constructor for extending the length of the chain and resume running it
   * Does a shallow copy of the provided chain and extending the member variable chainArray
   * @param chain Chain to be extended
   * @param nMoreSteps Number of steps to be extended
   */
  public AdaptiveRwmh(AdaptiveRwmh chain, int nMoreSteps) {
    //call superconstructor to do a shallow copy and extend the chain
    super(chain, nMoreSteps);
    //shallow copy member variables
    this.nStepTillAdaptive = chain.nStepTillAdaptive;
    this.e = chain.e;
  }
  
  
  /**OVERRIDE: STEP
   * Do a Metropolis-Hastings step, the chain is homogeneous for nStepTillAdaptive steps
   * Afterwards the proposal covariance will change, the step will be taken using
   * the method adaptiveStep
   * @param currentStep Column vector of the current step of the MCMC, to be modified
   */
  @Override
  public void step(SimpleMatrix currentPosition) {
    if (this.nStep < this.nStepTillAdaptive){
      this.metropolisHastingsStep(currentPosition); //homogeneous Metropolis-Hastings step
    } else {
      this.adaptiveStep(currentPosition); //adaptive Metropolis-Hastings step
    }
    this.updateStatistics(currentPosition);
  }
  
  
  /**METHOD: ADAPTIVE STEP
   * Do a Metropolis-Hastings step, proposal covariance will change according to the chain sample
   * covariance with a small element added to the diagonal
   * @param currentStep Column vector of the current step of the MCMC, to be modified
   */
  public void adaptiveStep(SimpleMatrix currentPosition) {
    //get the chain covariance and scale it so that it is optimial for targetting Normal
    SimpleMatrix proposalCovariance = new SimpleMatrix(this.chainCovariance);
    CommonOps_DDRM.scale(Math.pow(2.38, 2)/this.getNDim(), proposalCovariance.getDDRM());
    //add a small diagonal element to make the chain covariance full rank
    SimpleMatrix diagElement = SimpleMatrix.identity(this.getNDim());
    CommonOps_DDRM.scale(this.e, diagElement.getDDRM());
    CommonOps_DDRM.addEquals(proposalCovariance.getDDRM(), diagElement.getDDRM());
    //do a Metropolis-Hastings step with this proposal covariance
    this.proposalCovarianceChol = Global.cholesky(proposalCovariance);
    super.metropolisHastingsStep(currentPosition);
  }
}
