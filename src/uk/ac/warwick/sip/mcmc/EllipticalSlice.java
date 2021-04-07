/*
 *    Copyright 2018-2021 Sherman Lo

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

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

/**CLASS: ELLIPTICAL SLICE SAMPLER
 * Sampler which uses elliptical slice sampling
 * Reference: Elliptical slice sampling (Murray, I., Adams, R. P., and MacKay, D. J. (2010).
 *   Elliptical slice sampling. In Proceedings of the 13th International Conference on Artificial
 *   Intelligence and Statistics.)
 * At a step, a sample from the prior is taken. A weighted average between the current step and that
 *   sample is used as the next step in the MCMC according to the slice sampling scheme.
 */
public class EllipticalSlice extends Mcmc{

  //the likelihood of the model, elliptical slice sampling works on the likelihood rather than the
    //posterior
  protected TargetDistribution likelihood;
  //the prior distribution
  protected NormalDistribution prior;
  //position vector of the start of the ellipse (the current step)
  protected SimpleMatrix ellipse0;
  //position vector of the end of the ellipse (the sample from the prior)
  protected SimpleMatrix proposal;
  //after a step, an array of points looked at on the ellipse
  protected ArrayList<SimpleMatrix> ellipticalPositions;
  //slice variable, log uniform number plus log likelihood
  protected double sliceVariable;

  /**CONSTRUCTOR
   * Elliptical slice sampling algorithm. Samples from the prior (called the proposal in this code)
   *   and then uses slice sampling to find a weighted combination of the proposed and the current
   *   value, which turns out to make an ellipse.
   * @param target Object which has a method to call the pdf
   * @param likelihood the likelihood of the model
   * @param prior NormalDistribution representing the prior
   * @param chainLength Length of the chain to be obtained
   * @param rng Random number generator all the random numbers
   */
  public EllipticalSlice(TargetDistribution target, TargetDistribution likelihood,
      NormalDistribution prior, int chainLength, MersenneTwister rng) {
    super(target, chainLength, rng);
    this.likelihood = likelihood;
    this.prior = prior;
    this.ellipticalPositions = new ArrayList<SimpleMatrix>();
  }

  /**CONSTRUCTOR
   * Constructor for extending the length of the chain and resume running it
   * Does a shallow copy of the provided chain and extending the member variable chainArray
   * @param chain Chain to be extended
   * @param nMoreSteps Number of steps to be extended
   */
  public EllipticalSlice(EllipticalSlice chain, int nMoreSteps) {
    //shallow copies of member variables
    super(chain, nMoreSteps);
    this.likelihood = chain.likelihood;
    this.prior = chain.prior;
    this.proposal = chain.proposal;
    this.ellipticalPositions = chain.ellipticalPositions;
    this.sliceVariable = chain.sliceVariable;
  }

  @Override
  public void step(SimpleMatrix position) {
    //elliptical slice sampling is a weighted sum of position and a sample from the prior
    this.ellipse0 = new SimpleMatrix(position); //start of ellipse
    this.proposal = this.prior.sample(this.rng); //end of ellipse
    //array to store all points looked at on the ellipse
    this.ellipticalPositions = new ArrayList<SimpleMatrix>();

    double angle = this.sampleAngle(0, 2*Math.PI);
    double angleMin = angle - 2*Math.PI;
    double angleMax = angle;

    //sliceVariable is compared with the likelihood at points on the ellipse
    this.sliceVariable = Math.log(this.rng.nextDouble()) - this.likelihood.getPotential(position);
    boolean gotSample = false;

    //look through the ellipse until got valid sample
    while (!gotSample) {
      position.set(this.getPointOnEllipse(angle));
      if (this.isValidPointOnEllipse(position)) {
        gotSample = true;
      } else {
        //else this is not a valid point, look elsewhere on the ellipse
        this.ellipticalPositions.add(new SimpleMatrix(position));
        if (angle < 0) {
          angleMin = angle;
        } else {
          angleMax = angle;
        }
        angle = this.sampleAngle(angleMin, angleMax);
      }
    }
    this.updateStatistics(position);
  }

  /**METHOD: IS VALID POINT ON ELLIPSE
   * Check if this position vector would be accepted under the slice sampling scheme. This would
   *   require the member variable sliceVariable initalised with the correct value.
   * @param position position vector (on the ellipse)
   * @return boolean if the position vector is valid under slice sampling
   */
  public boolean isValidPointOnEllipse(SimpleMatrix position) {
    return -this.likelihood.getPotential(position) > this.sliceVariable;
  }

  /**METHOD: GET POINT ON ELLIPSE
   * Return a weighted sum of initial and proposal, the weight using cosine and sine of angle
   * @param initial vector of current position of chain
   * @param proposal vector of sample from prior
   * @param angle of the ellipse
   * @return vector, weighted sum of initial and proposal
   */
  public SimpleMatrix getPointOnEllipse(double angle) {
    SimpleMatrix initialWeighted = new SimpleMatrix(this.ellipse0);
    SimpleMatrix proposalWeighted = new SimpleMatrix(this.proposal);

    //centre the position vectors at the prior mean
    CommonOps_DDRM.subtractEquals(initialWeighted.getDDRM(), this.prior.mean.getDDRM());
    CommonOps_DDRM.subtractEquals(proposalWeighted.getDDRM(), this.prior.mean.getDDRM());

    CommonOps_DDRM.scale(Math.cos(angle), initialWeighted.getDDRM());
    CommonOps_DDRM.scale(Math.sin(angle), proposalWeighted.getDDRM());

    CommonOps_DDRM.addEquals(proposalWeighted.getDDRM(), initialWeighted.getDDRM());
    CommonOps_DDRM.addEquals(proposalWeighted.getDDRM(), this.prior.mean.getDDRM());

    return proposalWeighted;
  }

  /**METHOD: GET ELLIPTICAL POSITIONS ITERATOR
   * Return an iterator which iterates through all the points looked at on the ellipse
   */
  public Iterator<SimpleMatrix> getEllipticalPositionsIterator(){
    return this.ellipticalPositions.iterator();
  }

  /**METHOD: SAMPLE ANGLE`
   * Sample from the uniform distribution
   * @param min minimun value
   * @param max maximum value
   */
  protected double sampleAngle(double min, double max) {
    double angle = this.rng.nextDouble();
    angle *= max - min;
    angle += min;
    return angle;
  }

}
