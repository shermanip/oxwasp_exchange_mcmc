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

import org.ejml.simple.SimpleMatrix;

/**CLASS: UNIFORM DISTRIBUTION
 * Evaluates the uniform distribution pdf (up to a constant).
 * Use the method getPdf to evaluate the pdf.
 *
 */
public class UniformDistribution extends TargetDistribution{

  /**CONSTRUCTOR
   * @param nDim Number of dimensions
   */
  public UniformDistribution(int nDim) {
    super(nDim);
  }

  /**IMPLEMENT: GET PDF
   * Evaluate the probability density function at x
   * The pdf needs not to be normalised, ie integrate to 1
   * The normalisation constant is not needed for Metropolis-Hastings
   * @param x Where to evaluate the pdf, column vector
   * @return The evaluation of the pdf at x up to a constant
   */
  @Override
  public double getPdf(SimpleMatrix x) {
    return 1.0;
  }

  /**IMPLEMENT: GET POTENTIAL
   * Evaluate the -ln pdf + some constant
   * The constant comes from the face the pdf is evaluated up to a constant
   * @param x Where to evaluate the potential
   * @return The evaluation of the potential at x + some constant
   */
  @Override
  public double getPotential(SimpleMatrix x) {
    return 0.0;
  }

  /**METHOD: GET D POTENTIAL
   * Evaluate the differential of -ln pdf
   * @param x Where to evaluate the potential gradient
   * @return The evaluation of the potential gradient at x
   */
  @Override
  public SimpleMatrix getDPotential(SimpleMatrix x) {
    //return a vector of zeros
    return new SimpleMatrix(this.nDim, 1);

  }

}
