/*
 *    Copyright 2018-2020 Sherman Lo

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

/**CLASS: MIXTURE ADAPTIVE RANDOM WALK METROPOLIS HASTINGS
 * Adapts the proposal covariance using a mixture of homogeneous rwmh and scaled chain covariance
 * optimal for normal target
 * Reference: Gareth, O. and Rosenthal, R.S. (2009)
 * The adaptive procedure is as follows
 *   -2*this.getNDim()-1 initial steps are homogeneous
 *   -Afterwards the proposal covariance is then a scale of the chain sample covariance
 */
public class MixtureAdaptiveRwmh extends AdaptiveRwmh{

  protected SimpleMatrix safteyProposalCovarianceChol; //proposal covariance of the saftey step
  protected double probabilitySafety = 0.05; //probability of using the step

  /**CONSTRUCTOR
   * Adaptives the proposal covariance using a mixture of homogeneous rwmh and scaled chain
   * covariance optimal for normal target
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
    this.probabilitySafety = chain.probabilitySafety;
    this.safteyProposalCovarianceChol = chain.safteyProposalCovarianceChol;
  }

  /**OVERRIDE: ADAPTIVE STEP
   * Do a Metropolis-Hastings step but with adaptive proposal covariance
   * this.probabilitySaftey chance the proposal covarinace is safteyProposalCovarianceChol
   * Otherwise the proposal covariance is a scaled chain sample covariance
   * @param currentStep Column vector of the current step of the MCMC, to be modified
   */
  @Override
  public void adaptiveStep(SimpleMatrix currentStep) {

    //get the chain covariance and scale it so that it is optimial for targetting Normal
    SimpleMatrix newProposalCovarianceChol;
    newProposalCovarianceChol = new SimpleMatrix(this.chainCovariance);
    CommonOps_DDRM.scale(Math.pow(2.38, 2)/this.getNDim(), newProposalCovarianceChol.getDDRM());
    //Global.cholesky will return a null if the decomposition is unsuccessful
    //use the default proposal if a null is caught
    newProposalCovarianceChol = Global.cholesky(newProposalCovarianceChol);
    if (newProposalCovarianceChol == null) {
      newProposalCovarianceChol = this.safteyProposalCovarianceChol;
    }

    //with this.probabilitySaftey chance, use the saftey proposal covariance
    //else use the new proposal
    boolean isSaftey = this.rng.nextDouble()< this.probabilitySafety;
    if (isSaftey) {
      this.proposalCovarianceChol = this.safteyProposalCovarianceChol;
    }
    else {
      this.proposalCovarianceChol = newProposalCovarianceChol;
    }
    //do a Metropolis-Hastings step with this proposal covariance
    this.metropolisHastingsStep(currentStep);
    //save the new proposal
    if (isSaftey) {
      this.proposalCovarianceChol = newProposalCovarianceChol;
    }
  }

  /**METHOD: SET PROBABILITY SAFTEY
   * Set the probability that the proposal covariance is the safety proposal
   * @param probabilitySafety probability that the proposal covariance is the safety proposal
   */
  public void setProbabilitySaftey(double probabilitySafety) {
    this.probabilitySafety = probabilitySafety;
  }

}
