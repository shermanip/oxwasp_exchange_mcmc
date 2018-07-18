package uk.ac.warwick.sip.mcmcccfe;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.CholeskyDecomposition_F64;
import org.ejml.simple.SimpleMatrix;

public class Global {
	
	/**FUNCTION: CHOLESKY DECOMPOSITION
	 * @param x Symmetric matrix to decompose
	 * @return cholesky decomposition if possible, other null
	 */
	public static SimpleMatrix cholesky(SimpleMatrix x) {
		CholeskyDecomposition_F64<DMatrixRMaj> chol = DecompositionFactory_DDRM.chol(x.numRows(),true);
		if( !chol.decompose(x.getMatrix())) {
			return null;
		}
		return SimpleMatrix.wrap(chol.getT(null));
	}
	
	
	/**FUNCTION: GET RANDOM COVARIANCE
	 * Generates a random covariance using ZZ' where Z is a dxd matrix of random standard Gaussian
	 * @param nDim Number of dimensions
	 * @param rng MersenneTwister object to generate random numbers
	 * @return SimpleMatrix containing the covariance
	 */
	public static SimpleMatrix getRandomCovariance(int nDim, MersenneTwister rng) {
		SimpleMatrix covariance = new SimpleMatrix(nDim, nDim);
		for (int i=0; i<covariance.getNumElements(); i++) {
			covariance.set(i, rng.nextGaussian());
		}
		return covariance.mult(covariance.transpose());
	}

}
