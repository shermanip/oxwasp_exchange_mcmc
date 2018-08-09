package uk.ac.warwick.sip.mcmc;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

/**CLASS: RANDOM WALK METROPOLIS HASTINGS
 * Runs the metropolis hastings algorithm to sample a provided target distribution
 * The target distribution is provided via the constructor.
 * The chain length and a MersenneTwister is to be provided via the constructor.
 * Run the Metropolis Hastings algorithm using the step method, this will be one iteration
 * The samples are stored in the member variable chainArray as a design matrix format
 */
public class RandomWalkMetropolisHastings extends Mcmc{
  
  /**CONSTRUCTOR
   * Metropolis Hastings algorithm which targets a provided distribution using Gaussian random walk
   * @param target Object which has a method to call the pdf
   * @param chainLength Length of the chain to be obtained
   * @param rng Random number generator all the random numbers
   */
  public RandomWalkMetropolisHastings(TargetDistribution target, int chainLength,
      MersenneTwister rng){
    //assign member variables
    super(target, chainLength, rng);
  }
  
  /**CONSTRUCTOR
   * Constructor for extending the length of the chain and resume running it
   * Does a shallow copy of the provided chain and extending the member variable chainArray
   * @param chain Chain to be extended
   * @param nMoreSteps Number of steps to be extended
   */
  public RandomWalkMetropolisHastings(RandomWalkMetropolisHastings chain, int nMoreSteps){
    //do a shallow copy of chain
    super(chain, nMoreSteps);
  }
  
  /**IMPLEMENTED: STEP
   * This chain does a Metropolis-Hastings step
   * Instances created using RandomWalkMetropolisHastings will throw an exception because a
   * proposal covariance needs to be provided at every step
   * Subclasses should override this if a proposal covariance isn't required at every step
   */
  @Override
  public void step() {
    throw new RuntimeException("RandomWalkMetropolisHastings requires a proposal covariance in"
        + " the method step()");
  }
  
  /**METHOD: STEP
   * This chains takes a Metropolis-Hastings step and updates it member variables-
   * @param proposalCovarianceChol lower cholesky decomposition of the proposal_covariance
   */
  public void step(SimpleMatrix proposalCovarianceChol){
    this.metropolisHastingsStep(proposalCovarianceChol);
    this.updateStatistics();
  }
  
  /**METHOD: METROPOLIS HASTINGS STEP
   * Does a Metropolis-Hastings step
   * Saves the new value to the chain array given a proposal covariance
   * It does not increment nStep when this method is called
   * The method updateStatistics will increment nStep
   * @param proposalCovarianceChol lower cholesky decomposition of the proposal_covariance
   */
  protected void metropolisHastingsStep(SimpleMatrix proposalCovarianceChol){
    
    //instantiate column vector for the current value of the chain
    SimpleMatrix x = this.chainArray.extractVector(true, this.nStep);
    CommonOps_DDRM.transpose(x.getDDRM());
    
    //instantiate vector of N(0,1) using rng
    SimpleMatrix z = new SimpleMatrix(this.getNDim(), 1);
    for (int i=0; i<this.getNDim(); i++) {
      z.set(i, this.rng.nextGaussian());
    }
    
    //transform z using proposalCovarianceChol and x, assign it to y, y is a proposal
    SimpleMatrix y = proposalCovarianceChol.mult(z);
    CommonOps_DDRM.addEquals(y.getDDRM(), x.getDDRM());
    
    //declare variable for the acceptance probability, work it out using the ratio of target pdf
    //if it larger than one, then an acceptance step will always be taken
    double acceptProb = (this.target.getPdf(y)) / (this.target.getPdf(x));
    this.acceptStep(acceptProb, x, y);
  }
  
}
