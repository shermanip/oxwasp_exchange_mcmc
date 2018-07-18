package uk.ac.warwick.sip.mcmcccfe;

import java.lang.Math;

import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.decomposition.TriangularSolver_DDRM;
import org.ejml.simple.SimpleMatrix;

/**CLASS: NORMAL DISTRIBUTION
 * Evaluates the multivariate Normal distribution pdf (up to a constant).
 * The covariance is provided via the constructor.
 * Use the method getPdf to evaluate the pdf.
 *
 */
public class NormalDistribution extends TargetDistribution{
  
  //mean vector of the normal distribution
  protected SimpleMatrix mean;
  //cholesky decomposition of the covariance matrix as a lower triangle matrix
  protected SimpleMatrix covarianceChol;
  
  /**CONSTRUCTOR
   * Stores the covariance of the Normal random variable with mean 0
   * Returns the pdf when method getPdf is called
   * @param nDim Number of dimensions
   * @param covariance Covariance of the Normal random variable
   */
  public NormalDistribution(int nDim, SimpleMatrix mean, SimpleMatrix covariance){
    //assign member variables
    super(nDim);
    this.mean = mean;
    this.covarianceChol = Global.cholesky(covariance);
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
    return Math.exp(-this.getPotential(x));
  }
  
  /**IMPLEMENT: GET POTENTIAL
   * Evaluate the -ln pdf + some constant
   * The constant comes from the face the pdf is evaluated up to a constant
   * @param x Where to evaluate the potential
   * @return The evaluation of the potential at x + some constant
   */
  @Override
  public double getPotential(SimpleMatrix x) {
    SimpleMatrix z = this.covarianceChol.solve(x.minus(this.mean));
    return 0.5 * z.dot(z);
  }
  
  /**METHOD: GET D POTENTIAL
   * Evaluate the differential of -ln pdf
   * @param x Where to evaluate the potential gradient
   * @return The evaluation of the potential gradient at x
   */
  @Override
  public SimpleMatrix getDPotential(SimpleMatrix x) {
    //copy the memory in this,covarianceChol to covarianceCholInverse
    SimpleMatrix covarianceCholInverse = new SimpleMatrix(this.covarianceChol);
    //inverse covarianceCholInverse
    TriangularSolver_DDRM.invertLower(covarianceCholInverse.getDDRM().data, this.getNDim());
    //do the operation L^(T-1) * L^(-1) * x
    SimpleMatrix covariance = new SimpleMatrix(this.nDim, this.nDim);
    CommonOps_DDRM.multInner(covarianceCholInverse.getDDRM(), covariance.getDDRM());
    return covariance.mult(x.minus(this.mean));
    
  }
  
}
