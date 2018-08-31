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
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

public class BiasAdaptiveRwmh extends MixtureAdaptiveRwmh{
	
	/**CONSTRUCTOR
	 * Adaptives the proposal covariance using a mixture of homogeneous rwmh and scaled chain
	 * covariance optimal for normal target
	 * @param target See superclass RandomWalkMetropolisHastings
	 * @param chainLength See superclass RandomWalkMetropolisHastings
	 * @param proposalCovariance proposal covariance use in homogeneous steps
	 * @param rng See superclass RandomWalkMetropolisHastings
	 */
	public BiasAdaptiveRwmh (TargetDistribution target, int chainLength,
			SimpleMatrix proposalCovariance, MersenneTwister rng){
		super(target, chainLength, proposalCovariance, rng);
	}
	
	/**CONSTRUCTOR
	 * Adaptives the proposal covariance using a mixture of homogeneous rwmh and scaled chain
	 * covariance optimal for normal target
	 * @param target See superclass RandomWalkMetropolisHastings
	 * @param chainLength See superclass RandomWalkMetropolisHastings
	 * @param proposalCovariance proposal covariance use in homogeneous steps
	 * @param rng See superclass RandomWalkMetropolisHastings
	 */
	public BiasAdaptiveRwmh(BiasAdaptiveRwmh chain, int nMoreSteps) {
		super(chain, nMoreSteps);
	}
	
	/**METHOD: SET INITIAL VALUE
	 * Set the initial value of the chain, to be called before running the chain
	 * Calling this will initalise the member variable chainMean
	 * @param initialValue double [] containing the values of the initial position, needs to be of
	 * of the correction dimensions
	 */
	@Override
	public void setInitialValue(double [] initialValue) {
		super.setInitialValue(initialValue);
		CommonOps_DDRM.divide(this.chainMean.getDDRM(), 2.0);
		
		//x1 is zero
		//get the initial value of the chain and transpose it to be a column vector
		SimpleMatrix x1 = this.chainArray.extractVector(true, 0);
		CommonOps_DDRM.transpose(x1.getDDRM());
		
		SimpleMatrix x0 = new SimpleMatrix(this.getNDim(), 1);
		
		//calculate the difference between the initial and the mean
		SimpleMatrix r1 = x1.minus(this.chainMean);
		//calculate the difference between the most recent value and the mean
		SimpleMatrix r0 = x0.minus(this.chainMean);
		//calculate the covariance from scratch
		CommonOps_DDRM.multOuter(r1.getDDRM(), this.chainCovariance.getDDRM());
		DMatrixRMaj r0Outer = new DMatrixRMaj(this.getNDim(), this.getNDim());
		CommonOps_DDRM.multOuter(r0.getDDRM(), r0Outer);
		CommonOps_DDRM.addEquals(this.chainCovariance.getDDRM(), r0Outer);
		
	}
	
	/**METHOD: UPDATE STATISTICS
	 * Update the acceptanceArray, nStep, chainMean and chainCovariance
	 * @param x The new position column vector of the chain, after the MCMC step(s)
	 */
	@Override
	protected void updateStatistics(SimpleMatrix x){
		
		//work out the acceptance rate at this stage
		//acceptance rate = (number of acceptance steps) / (number of steps)
		//the acceptance rate keeps track of the acceptance rate from and including the 1st step
		//(not from the initial value)
		this.acceptanceArray[this.nSample-1] = ((double)(this.nAccept)) / ((double)(this.nStep+1));
		
		//increment the number of steps taken
		this.nStep++;
		//n is the chain length (for no thinning)
		//so it is the number of steps + 1 (from the initial value)
		double n = (double) (this.nStep+2);
		
		//update the mean using the previous mean
		CommonOps_DDRM.scale(n-1, this.chainMean.getDDRM());
		CommonOps_DDRM.addEquals(this.chainMean.getDDRM(), x.getDDRM());
		CommonOps_DDRM.divide(this.chainMean.getDDRM(), n);
		
		//calculate the difference between the most recent value and the mean
		SimpleMatrix r2 = x.minus(this.chainMean);
		//update the covariance using the previous covariance
		CommonOps_DDRM.scale(n-2, this.chainCovariance.getDDRM());
		DMatrixRMaj r2Outer = new DMatrixRMaj(this.getNDim(), this.getNDim());
		CommonOps_DDRM.multOuter(r2.getDDRM(), r2Outer);
		CommonOps_DDRM.scale(n/(n-1), r2Outer);
		CommonOps_DDRM.addEquals(this.chainCovariance.getDDRM(), r2Outer);
		CommonOps_DDRM.divide(this.chainCovariance.getDDRM(), n-1);
		
	}
}
