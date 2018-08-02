package uk.ac.warwick.sip.mcmcccfe;

import java.util.ArrayList;
import org.apache.commons.math3.stat.inference.OneWayAnova;
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
  
  /**METHOD: GET GELMAN RUBIN F ARRAY
   * Calculates the F statistic for different nBurnIn 2,3,...,maxNBurnIn
   * See method getGelmanRubinF for description of the F statistic
   * @param maxNBurnIn Maxmimum number of samples to burn in to be investigated
   * @return array of f statistics for nBurnIn 2,3,...,maxNBurnIn
   */
  public double [] getGelmanRubinFArray(int maxNBurnIn) {
    //declare array for F statistics, this doesn't include nBurnIn=1
    double [] fArray = new double[maxNBurnIn-1];
    //for each nBurnIn, get the F statistic and save it in the array
    for (int i=0; i<(maxNBurnIn-1); i++) {
      //calculate the F statistic for nBurnIn = 2,3,...,maxNBurnIn
      fArray[i] = this.getGelmanRubinF(i+2);
    }
    return fArray;
  }
  
  /**METHOD: GET GELMAN RUBIN F STATISTIC
   * Calculates the F statistic for a given nBurnIn
   * The F statistics is calculated using the values in chain [nBurnIn : 2*nBurnIn - 1];
   * In other words, we look at the nBurnIn samples after burn in
   * Using multiple chains provided, a One-Way ANOVA is conducted using these burnt in samples
   * The One-Way ANOVA returns a ratio of the variance between chains over the variance within 
   * chains, this is the F statistic
   * @param nBurnIn The number of samples at the start of the chain to be ignored
   * @return Gelman's F statistic, ANOVA version
   */
  protected double getGelmanRubinF(int nBurnIn) {
    //declare array for storing double []
    //these double [] represent the FULL chain
    ArrayList <double []> chainArrayList = new ArrayList <double []>();
    double [] chain; //declare array for a chain
    double [] chainBurnIn; //declare array for a burnin chain
    //for each chain
    for (int iChain=0; iChain<this.chainArray.size(); iChain++) {
      //get the chain
      chain = this.chainArray.get(iChain);
      //copy the values of the burnt values
      chainBurnIn = new double[nBurnIn];
      for (int i=0; i<nBurnIn; i++) {
        chainBurnIn[i] = chain[nBurnIn+i];
      }
      //add the burnt in chain to the array
      chainArrayList.add(chainBurnIn);
    }
    //using the array of chains, do a one way anova and return the F statistic
    OneWayAnova anova = new OneWayAnova();
    return anova.anovaFValue(chainArrayList);
    
  }
  
}
