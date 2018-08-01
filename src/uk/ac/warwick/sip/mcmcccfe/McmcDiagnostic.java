package uk.ac.warwick.sip.mcmcccfe;

import java.util.ArrayList;
import org.ejml.simple.SimpleMatrix;

public class McmcDiagnostic {
  
  //array of MCMC, mcmc chains are represented as a double [], each entry for each step
  protected ArrayList <double[]> chainArray = new ArrayList<double[]>();
  
  /**CONSTRUCTOR
   * Empty constructor
   */
  public McmcDiagnostic() {
  }
  
  /**METHOD: ADD CHAIN
   * Add a MCMC chain to this object
   * @param chain values of a MCMC chain to be added to this object
   */
  public void addChain(double[] chain) {
    this.chainArray.add(chain);
  }
  
  /**METHOD: GET AUTOCORRELATION FUNCTION
   * Calculates the sample autocorrelation function for lags 0 to nLag
   * Results are returned in a double []
   * @param nLag The maximum lag to be obtained
   * @return The acf at lag 0, 1, 2, ..., nLag
   */
  public double [] getAcf(int nLag) {
    
    //declare array for the acf, for lag 0,1,2,...,nLag
    double [] acf = new double[nLag+1];
    //retrieve the chain
    double [] chain = this.chainArray.get(0);
    //wrap the chain in a SimpleMatrix
    //(uses a constructor which copies the content of the double [])
    SimpleMatrix chainVector = new SimpleMatrix(chain.length, 1, true, chain);
    
    //work out the sample mean and centre the chain at the sample mean
    double mean = chainVector.elementSum() / chain.length;
    chainVector = chainVector.minus(mean);
    //work out the S_x_xLag for all lags, see method getSxxlag
    for (int i=0; i<=nLag; i++) {
      acf[i] = this.getSxxlag(chainVector, i);
    }
    //normalise the acf
    for (int i=1; i<=nLag; i++) {
      acf[i] /= acf[0];
    }
    acf[0] = 1.0;
    
    return acf;
    
  }
  
  /**METHOD: GET S_X_XLAG
   * Calculates \sum_{i=0}^{n-k} (x_i)(x_{i+k}) where k is the lag
   * For k = 0, this is the sum of squares
   * @param chain Column vector containing the chain, centred around the mean
   * @param lag k
   * @return Sum of lagged elements
   */
  protected double getSxxlag(SimpleMatrix chain, int lag) {
    //trim the front of the chain
    SimpleMatrix chainFrontTrim = chain.extractMatrix(lag, chain.numRows(), 0, 1);
    //trim the back of the chain
    SimpleMatrix chainEndTrim = chain.extractMatrix(0, chain.numRows()-lag, 0, 1);
    //multiply the trimmed chains
    return chainFrontTrim.elementMult(chainEndTrim).elementSum();
    
  }
  
}
