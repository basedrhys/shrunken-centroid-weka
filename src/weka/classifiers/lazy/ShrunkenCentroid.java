/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ShrunkenCentroid.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.lazy;

import org.jetbrains.annotations.NotNull;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.util.Arrays;

/**
 <!-- globalinfo-start -->
 * Enhancement of a simple centroid classifier, this algorithm shrinks the centroids toward the global centroid
 * to improve accuracy.
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Diagnosis of multiple cancer types by shrunken centroids of gene expression
 * Robert Tibshirani, Trevor Hastie, Balasubramanian Narasimhan, Gilbert Chu
 * Proceedings of the National Academy of Sciences May 2002, 99 (10) 6567-6572; DOI: 10.1073/pnas.082099299
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{tibshirani2002diagnosis,
 *   title={Diagnosis of multiple cancer types by shrunken centroids of gene expression},
 *   author={Tibshirani, Robert and Hastie, Trevor and Narasimhan, Balasubramanian and Chu, Gilbert},
 *   journal={Proceedings of the National Academy of Sciences},
 *   volume={99},
 *   number={10},
 *   pages={6567--6572},
 *   year={2002},
 *   publisher={National Acad Sciences}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -D
 *  Delta -- shrinkage parameter, how much to shrink the centroids towards the global centroid.</pre>
 *
 <!-- options-end -->
 *
 * @author Rhys Compton (rhys.compton@gmail.com)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */

public class ShrunkenCentroid extends AbstractClassifier {

    @OptionMetadata(
            displayName = "Delta",
            description = "Shrinkage parameter",
            commandLineParamName = "D",
            commandLineParamSynopsis = "-D <double>",
            displayOrder = 1)
    public double getDelta() {
        return m_delta;
    }

    public void setDelta(int d) {
        m_delta = d;
    }

    // Main hyperparameter, controls how much shrinkage to perform on the centroids
    // Will be automatically found through CV in the future
    protected double m_delta = 0.2;

    // The global centroid -- average of *all* instances in the training set
    private Centroid m_globalCentroid;

    // Class level centroids -- average of all instances for each class, in order of class values
    private Centroid[] m_classCentroids;

//    // Number of attributes for the centroids
//    private int m_centroidNumAttributes;

    private int m_classAttributeIndex;

    // Within-class standard deviation for all attributes (for all i)
    private double[] m_withinClassStdDevSi;

    //
    private double[] m_allMK;

    // Median of all the within-class standard deviations, 'So' in the paper formulas
    private double m_medianSo;

    // Collection of t statistics for attribute i, comparing class k to the overall centroid,
    // 'dik' in the paper formulas
    private double[][] m_tStatisticsDik;

    // Set to false to create a standard Nearest Centroid Classifier -- no shrinkage
    protected boolean doShrinkage = true;

    public void buildClassifier(Instances trainingData) {
        trainingData = new Instances(trainingData);
        trainingData.deleteWithMissingClass();

        // Create the global and class centroids
        calculateCentroids(trainingData);

        if (doShrinkage) {
            // Calculate Si for each i (for each attribute)
            calculateStandardDeviations(trainingData);
            // Calculate So (simply use the median of the Si values
            calculateMedian(m_withinClassStdDevSi);
            // Calculate Mk for all classes
            calculateStandardizingParamsForAllK(trainingData);

            // Calculate d'ik for all i and k (all attributes and class centroids)
            calculateAllTStatistics();

            // Shrink the centroids
            shrinkCentroids();
        }
    }

    private void shrinkCentroids() {
        // Using the values calculated, we finally shrink the centroids
        // equation 4 in the paper
        for (int k = 0; k < m_classCentroids.length; k++) {

            // Get this class centroid and class Mk
            Centroid classCentroid = m_classCentroids[k];
            double thisMk = m_allMK[k];

            for (int i = 0; i < m_globalCentroid.numAttributes(); i++) {
                // Ignore the class attribute
                if (i == m_classAttributeIndex)
                    continue;
                //x(hat)i + mk(si + so)d'ik
                double newAttrValue = m_globalCentroid.getValue(i) + thisMk * (m_withinClassStdDevSi[i] + m_medianSo) * m_tStatisticsDik[i][k];
                classCentroid.setValue(i, newAttrValue);
            }
        }
    }

    private void calculateAllTStatistics() {
        // Make a 2d array with i rows (i attributes) and k columns (k classes)
        m_tStatisticsDik = new double[m_globalCentroid.numAttributes()][m_classCentroids.length];

        // Now iterate over all attributes and calculate the t statistic for each class
        // (equation 1 in the paper, for calculating dik, and equation 5 for d'ik)
        for (int k = 0; k < m_classCentroids.length; k++) {

            // Get this class centroid and class Mk
            Centroid classCentroid = m_classCentroids[k];
            double thisMk = m_allMK[k];

            for (int i = 0; i < m_globalCentroid.numAttributes(); i++) {
                // Ignore the class attribute
                if (i == m_classAttributeIndex)
                    continue;
                // Top half of equation 1
                double difFromGlobal = m_globalCentroid.getDifferenceFromInstanceAttribute(classCentroid.getInstance(), i);
                // Bottom half of equation 1
                double stdError = thisMk * (m_withinClassStdDevSi[i] + m_medianSo);
                // Finish equation 1
                double dik = difFromGlobal / stdError;

                // Now calculate d'ik - equation 5
                double dPrime = getDPrime(dik);
                // Save the value
                m_tStatisticsDik[i][k] = dPrime;
            }
        }
    }

    private double getDPrime(double dik) {
        // Equation 5 in the paper

        // Get the absolute difference between the value and delta
        double difference  = Math.abs(dik) - m_delta;
        // Only keep the positive part
        if (difference < 0) {
            difference = 0;
        }
        // Add the sign back in
        int sign = dik < 0 ? -1 : 1;
        difference *= sign;
        // Equation 5 finished
        return difference;
    }

    private void calculateStandardizingParamsForAllK(Instances trainingData) {
        // Calculate an mk for each class, just after equation 2
        m_allMK = new double[m_classCentroids.length];
        // TODO more meaningful name
        double oneOverAll = 1f / trainingData.numInstances();

        // Loop over all classes
        for (int i = 0; i < m_classCentroids.length; i++) {
            // Get the relevant class centroid
            Centroid c = m_classCentroids[i];
            // Calculate the first half of the equation (1/nk) TODO give more meaningful names
            double firstEq = 1f / c.getInstances().size();
            // Calculate the actual Mk
            double mk = Math.sqrt(firstEq + oneOverAll);
            // Save the value
            m_allMK[i] = mk;
        }
    }

    private void calculateMedian(double[] values) {
        // Copy it as we need to sort the values to find median
        double[] valuesCopy = values.clone();
        Arrays.sort(valuesCopy);

        int middle = valuesCopy.length / 2;
        if (valuesCopy.length % 2 == 1) {
            m_medianSo = valuesCopy[middle];
        } else {
            m_medianSo = (valuesCopy[middle-1] + valuesCopy[middle]) / 2.0;
        }
    }

    private void calculateStandardDeviations(@NotNull Instances trainingData) {
        // Equation 2 in the paper, calculate within-class std dev
        m_withinClassStdDevSi = new double[m_globalCentroid.numAttributes()];
        // 1 / n - K
        double stdValue = 1 / (float) (trainingData.numInstances() - m_classCentroids.length);

        // We want to calculate Si for all i
        for (int i = 0; i < m_globalCentroid.numAttributes(); i++) {
            // Ignore the class attribute
            if (i == m_classAttributeIndex)
                continue;
            double sum = 0;
                // Outer sum : for all classes
                for (Centroid thisClassCentroid : m_classCentroids) {
                    // Inner sum : for all instances in this class
                    for (Instance instance : thisClassCentroid.getInstances()) {
                        // Xij - X-ik - difference of this attribute between this instance and the class centroid
                        double dif = thisClassCentroid.getDifferenceFromInstanceAttribute(instance, i);
                        // Square the value
                        dif = dif * dif;
                        // Add it to the sum for this attribute
                        sum += dif;
                    }
                }
                // Multiply it by the standard value
                sum *= stdValue;
                // Square root it (equation 2 is for si^2, we want si)
                sum = Math.sqrt(sum);
                // Save the standard deviation for this attribute
                m_withinClassStdDevSi[i] = sum;
        }
    }

    private void calculateCentroids(@NotNull Instances trainingData) {
//        m_centroidNumAttributes = trainingData.numAttributes() - 1; // ignore class value for calculating centroids
        // Init the global and class centroids
        m_globalCentroid = new Centroid(trainingData.numAttributes());
        m_classAttributeIndex = trainingData.classIndex();
        m_classCentroids = new Centroid[trainingData.classAttribute().numValues()];
        for (int i = 0; i < m_classCentroids.length; i++) {
            m_classCentroids[i] = new Centroid(trainingData.numAttributes());
        }

        // For each instance, add them into the global and class centroid
        for (Instance instance : trainingData) {
            // Get the class value
            int classVal = (int) instance.classValue();

            m_globalCentroid.addInstance(instance);

            // Get the centroid for this class
            Centroid tempClassCentroid = m_classCentroids[classVal];
            // Add the value for this instance
            tempClassCentroid.addInstance(instance);
        }

        // After adding up all the attribute values, we finally average them to find the global and class centroid
        m_globalCentroid.averageValues();
        for (Centroid c : m_classCentroids) {
            c.averageValues();
        }
    }

    public double classifyInstance(Instance testInstance) {
        // Max value as we want all calculated distances to be less
        double minDist = Double.MAX_VALUE;
        double minDistClass = 0;
        if (doShrinkage) {
            // We need to scale the instances once we've shrunken the centroids
            // Here we use equation [6]
            // Find the centroid with the lowest distance
            for (int k = 0; k < m_classCentroids.length; k++) {
                Centroid classCentroid = m_classCentroids[k];
                double distanceSum = 0;
                // sum from i = 1 to p
                for (int i = 0; i < m_globalCentroid.numAttributes(); i++) {
                    // Ignore the class attribute
                    if (i == m_classAttributeIndex)
                        continue;
                    double squaredDif = classCentroid.getDifferenceFromInstanceAttribute(testInstance, i);
                    // Actually square it
                    squaredDif *= squaredDif;
                    // bottom half of eq - (si + so)^2
                    double standardizeVal = m_withinClassStdDevSi[i] + m_medianSo;
                    standardizeVal *= standardizeVal;
                    // Add this value into the sum over all attributes
                    double thisSum = squaredDif / standardizeVal;
                    distanceSum += thisSum;
                }
                // Last part of equation -- 2 log pi k
                // Proportion of instances in this class
                double classPrior = classCentroid.getInstances().size() / (float) m_globalCentroid.getInstances().size();
                double distanceCorrected = distanceSum - (2 * Math.log(classPrior));
                if (distanceCorrected < minDist) {
                    minDist = distanceCorrected;
                    minDistClass = k;
                }
            }
        } else {
            // Perform a standard nearest centroid classification if we're not doing shrinkage
            for (int i = 0; i < m_classCentroids.length; i++) {
                // Calculate the distance to each centroid
                Centroid classCentroid = m_classCentroids[i];
                double tmpDist = classCentroid.getDistanceFromInstance(testInstance);
                // Save the lowest distance and the class
                if (tmpDist < minDist) {
                    minDist = tmpDist;
                    minDistClass = i;
                }
            }
        }
        return minDistClass;
    }

    /**
     * The info shown in the GUI.
     * @return the info describing the filter.
     */
    public String globalInfo() {
        return "This algorithm performs nearest-centroid classification, however as an enhancement," +
                "shrinks the centroids toward the global centroid, controlled by the Delta parameter.";
    }

    /**
     * The capabilities of this filter.
     * @return the capabilities
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enable(Capabilities.Capability.STRING_ATTRIBUTES);
        result.enableAllClasses();
        return result;
    }
}
