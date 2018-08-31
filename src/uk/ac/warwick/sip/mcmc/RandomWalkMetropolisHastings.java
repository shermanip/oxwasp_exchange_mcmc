/*
 *    Copyright 2018 Sherman Ip

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

package uk.ac.warwick.sip.mcmc;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

/**CLASS: RANDOM WALK METROPOLIS HASTINGS
 * Runs the metropolis hastings algorithm to sample a provided target distribution
 * Reference: Metropolis, N., et.al. (1953), Hastings, W.K. (1970)
 * This implementation does a Gaussian random walk for the proposal, with covariance defined in the
 * member variable proposalCovarianceChol
 */
public class RandomWalkMetropolisHastings extends Mcmc{
  
  //proposal covariance decomposed using cholesky
  protected SimpleMatrix proposalCovarianceChol;
  
  /**CONSTRUCTOR
   * Metropolis Hastings algorithm which targets a provided distribution using Gaussian random walk
   * @param target Object which has a method to call the pdf
   * @param chainLength Length of the chain to be obtained
   * @param proposalCovariance proposal covariance, symmetric square matrix, size target.getNDim()
   * @param rng Random number generator all the random numbers
   */
  public RandomWalkMetropolisHastings(TargetDistribution target, int chainLength,
      SimpleMatrix proposalCovariance, MersenneTwister rng){
    //assign member variables
    super(target, chainLength, rng);
    this.proposalCovarianceChol = Global.cholesky(proposalCovariance);
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
    //shallow copy the proposalCovarianceChol
    this.proposalCovarianceChol = chain.proposalCovarianceChol;
  }
  
  /**IMPLEMENTED: STEP
   * This chains takes a Metropolis-Hastings step and updates it member variables
   * @param currentStep Column vector of the current step of the MCMC, to be modified
   */
  @Override
  public void step(SimpleMatrix currentPosition){
    this.metropolisHastingsStep(currentPosition);
    this.updateStatistics(currentPosition);
  }
  
  /**METHOD: METROPOLIS HASTINGS STEP
   * Does a Metropolis-Hastings step given a proposal covariance defined in proposalCovarianceChol
   * It does not increment nStep when this method is called
   * The method updateStatistics will increment nStep
   * @param x current position of the chain, to be modified
   */
  protected void metropolisHastingsStep(SimpleMatrix x){
    
    //instantiate vector of N(0,1) using rng
    SimpleMatrix z = new SimpleMatrix(this.getNDim(), 1);
    for (int i=0; i<this.getNDim(); i++) {
      z.set(i, this.rng.nextGaussian());
    }
    
    //transform z using proposalCovarianceChol and x, assign it to y, y is a proposal
    SimpleMatrix y = this.proposalCovarianceChol.mult(z);
    CommonOps_DDRM.addEquals(y.getDDRM(), x.getDDRM());
    
    //declare variable for the acceptance probability, work it out using the ratio of target pdf
    //if it larger than one, then an acceptance step will always be taken
    double acceptProb = (this.target.getPdf(y)) / (this.target.getPdf(x));
    this.acceptStep(acceptProb, x, y); //x can be modified here
  }
  
  /**METHOD: SET PROPOSAL COVARIANCE
   * Set the proposal covariance
   * @param proposalCovariance proposal covariance use in homogeneous steps
   */
  public void setProposalCovariance(SimpleMatrix proposalCovariance) {
    this.proposalCovarianceChol = Global.cholesky(proposalCovariance);
  }
  
}
