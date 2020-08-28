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

import java.util.ArrayList;

import org.apache.commons.math3.stat.inference.OneWayAnova;

/**CLASS: GELMAN RUBIN F STATISTIC
 * Diagnostic for juding the convergence of MCMC. This is done by investigating multiple chains,
 * with different starting points, and looking at the mean between and within chain, similar to
 * ANOVA.
 *
 * How to use: construct an array of Mcmc and run each and every chain.
 * Then call the getGelmanRubinFArray or getGelmanRubinF to obtain the statistic.
 *
 */
public class GelmanRubinF {

  //array of MCMC, mcmc chains are represented as a double [], each entry for each step
  protected Mcmc [] chainArray;

  /**CONSTRUCTOR
   * @param chainArray array of chains to be used for the Gelman Rubin statistic
   */
  public GelmanRubinF(Mcmc [] chainArray) {
    this.chainArray = chainArray;
  }

  /**METHOD: GET GELMAN RUBIN F ARRAY
   * Calculates the F statistic for different nBurnIn 2,3,...,maxNBurnIn
   * See method getGelmanRubinF for description of the F statistic
   * @param nDim which dimension to investigate
   * @param maxNBurnIn Maxmimum number of samples to burn in to be investigated
   * @return array of f statistics for nBurnIn 2,3,...,maxNBurnIn
   */
  public double [] getGelmanRubinFArray(int nDim, int maxNBurnIn) {
    //declare array for F statistics, this doesn't include nBurnIn=1
    double [] fArray = new double[maxNBurnIn-1];
    //for each nBurnIn, get the F statistic and save it in the array
    for (int i=0; i<(maxNBurnIn-1); i++) {
      //calculate the F statistic for nBurnIn = 2,3,...,maxNBurnIn
      fArray[i] = this.getGelmanRubinF(nDim, i+2);
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
   * @param nDim which dimension to investigate
   * @param nBurnIn The number of samples at the start of the chain to be ignored
   * @return Gelman's F statistic, ANOVA version
   */
  protected double getGelmanRubinF(int nDim, int nBurnIn) {
    //declare array for storing double []
    //these double [] represent the FULL chain
    ArrayList <double []> chainArrayList = new ArrayList <double []>(this.chainArray.length);
    double [] chain; //declare array for a chain
    double [] chainBurnIn; //declare array for a burnin chain
    //for each chain
    for (int iChain=0; iChain<this.chainArray.length; iChain++) {
      //get the chain
      chain = this.chainArray[iChain].getChain(nDim);
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
