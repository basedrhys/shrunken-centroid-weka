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

import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.*;

import java.util.*;

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
 *
 * @author Rhys Compton (rhys.compton@gmail.com)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */

public class ShrunkenCentroid extends AbstractClassifier {

    /*
        Getters and setters for main hyperparameters
     */
    // Main hyperparameter, controls how much shrinkage to perform on the centroids
    protected double m_shrinkageThreshold = 0.2;

    public double getShrinkage() {
        return m_shrinkageThreshold;
    }

    public void setShrinkage(double d) {
        m_shrinkageThreshold = d;
    }

    public String shrinkageTipText() {
        return "Controls how much shrinkage to perform on the centroids";
    }

    // The number of thresholds to evaluate during internal CV
    // Does not affect the min and max threshold, just how many
    // are evaluated in between
    protected int m_numEvaluationThresholds = 30;

    public int getNumEvaluationThresholds() {
        return m_numEvaluationThresholds;
    }

    public void setNumEvaluationThresholds(int m_numEvaluationThresholds) {
        this.m_numEvaluationThresholds = m_numEvaluationThresholds;
    }

    public String numEvaluationThresholdsTipText() { return "The number of thresholds to evaluate during internal CV. " +
            "Does not affect the min and max threshold, just how many" +
            " are evaluated in between"; }

    // Set to false to create a standard Nearest Centroid Classifier -- no shrinkage
    protected boolean m_applyShrinkage = true;

    public boolean getApplyShrinkage() {
        return m_applyShrinkage;
    }

    public void setApplyShrinkage(boolean applyShrinkage) {
        this.m_applyShrinkage = applyShrinkage;
    }

    public String applyShrinkageTipText() { return "Set to false to create a standard Nearest Centroid Classifier -- no shrinkage"; }

    /*
        Private Variables
     */
    private boolean m_inCV = false;

    // The global centroid -- average of *all* instances in the training set
    private Centroid m_globalCentroid;

    // Class level centroids -- average of all instances for each class, in order of class values
    private Centroid[] m_classCentroids;

    // Index of the class attribute (so we don't include it in centroid location calculations
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

    // Used to find maximum shrinkage threshold to evaluate
    private double m_maxTStatistic = -1;

    private final String SUMMARY_STRING = "\nBest values found through CV:\n" +
                                    "Threshold - %3f   |   Accuracy - %3f\n";

    /**
     * Parses a given list of options. <p/>
     * <p>
     * <!-- options-start -->
     * Valid options are: <p/>
     * <p>
     * <pre> -S &lt;shrinkage threshold&gt;
     *  Sets the threshold to shrink from - larger values will shrink more attributes to 0
     *  (default 0.5) </pre>
     * <p/>
     * <p>
     * <pre> -num-thresholds &lt;number of thresholds&gt;
     *  The number of thresholds to evaluate during internal CV. Does not affect the
     *  min and max threshold, just how many are evaluated in between
     *  (default 30) </pre>
     *  <p/>
     * <p>
     * <pre> -apply-shrinkage &lt;value&gt;
     *  If false, don't shrink the centroids, just create a standard nearest centroid classifier
     *  (default true)
     * </pre>
     * <p/>
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String shrinkage = Utils.getOption("S", options);
        // Set the attribute subset size if given
        if (shrinkage.length() != 0) {
            setShrinkage(Double.parseDouble(shrinkage));
        } else {
            setShrinkage(0.2);
        }

        String numThresholds = Utils.getOption("num-thresholds", options);
        if (numThresholds.length() != 0) {
            setNumEvaluationThresholds(Integer.parseInt(numThresholds));
        } else {
            setNumEvaluationThresholds(30);
        }

        String applyShrinkage = Utils.getOption("apply-shrinkage", options);
        if (applyShrinkage.length() != 0) {
            setApplyShrinkage(Boolean.parseBoolean(applyShrinkage));
        } else {
            setApplyShrinkage(true);
        }

        // Set all the other options
        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        String[] superOptions = super.getOptions();
        String[] options = new String[superOptions.length + 6];
        int current = 0;
        options[current++] = "-S";
        options[current++] = "" + getShrinkage();

        options[current++] = "-num-thresholds";
        options[current++] = "" + getNumEvaluationThresholds();

        options[current++] = "-apply-shrinkage";
        options[current++] = "" + getApplyShrinkage();
        System.arraycopy(superOptions, 0, options, current, superOptions.length);
        return options;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<>(3);
        newVector.addElement(new Option("\tSets how much shrinkage to apply to the centroids",
                "S", 1, "-S <shrinkage>"));

        newVector.addElement(new Option("\tThe number of thresholds to evaluate during internal CV",
                "num-thresholds", 1, "-num-thresholds <number of thresholds>"));

        newVector.addElement(new Option("\tIf false, don't shrink the centroids, just create a standard " +
                "nearest centroid classifier", "apply-shrinkage", 1,
                "-apply-shrinkage <value>"));

        newVector.addAll(Collections.list(super.listOptions()));
        return newVector.elements();
    }

    private void doDatasetCalculations(Instances trainingData) {
        trainingData = new Instances(trainingData);
        trainingData.deleteWithMissingClass();

        // Create the global and class centroids
        calculateCentroids(trainingData);

        // Calculate Si for each i (for each attribute)
        calculateStandardDeviations(trainingData);
        // Calculate So (simply use the median of the Si values
        calculateMedian(m_withinClassStdDevSi);
        // Calculate Mk for all classes
        calculateStandardizingParamsForAllK(trainingData);

        // Calculate d'ik for all i and k (all attributes and class centroids)
        calculateAllTStatistics();
    }

    private double[] calculateShrinkageThresholds() {
        float current = 0;
        float end = (float) m_maxTStatistic;
        float step = end / (getNumEvaluationThresholds() - 1); // -1 so the last threshold *is* the maximum t statistic

        double[] ret = new double[getNumEvaluationThresholds()];

        for (int i = 0; i < getNumEvaluationThresholds(); i++) {
            ret[i] = current;
            current+=step;
        }
        return ret;
    }

    private int getNonZeroAttributes() {
        int count = 0;
        for (Centroid c : m_classCentroids) {
            count += c.getNonZeroShrunkenAttributes();
        }
        return count;
    }

    public void buildClassifier(Instances trainingData) throws Exception {
        double bestPercent = -1;
        double bestThreshold = 0;
        // This ensures that the buildClassifier() function isn't called recursively forever
        // When buildClassifier() is called during the crossValidationModel() below,
        if (!m_inCV) {
            // Do the non-threshold calculations on the dataset
            // i.e. calculate overall centroids, standard deviations...
            doDatasetCalculations(trainingData);

            // We don't need to do anything more, we've already build the centroids so are ready for classification
            if (!m_applyShrinkage) {
                return;
            }

            // Calculate all the different shrinkage thresholds we'll try
            double[] thresholds = calculateShrinkageThresholds();

            m_inCV = true;

            Evaluation evaluation = new Evaluation(trainingData);
            for (double threshold : thresholds) {

                // Using this current threshold, shrink the centroids
                shrinkCentroids(threshold);

                // Perform 10 fold CV using the shrunken centroids
                evaluation.crossValidateModel(this, trainingData, 10, new Random(1));
                double pctCorrect = evaluation.pctCorrect();
                int nonZero = getNonZeroAttributes();
                    System.out.printf("%3f --- %3f\n", threshold, evaluation.pctCorrect());
                }
                // If this is better than the previous best, set this as the new best
                // threshold
                if (pctCorrect > bestPercent) {
                    bestThreshold = threshold;
                    bestPercent = pctCorrect;
                    if (m_Debug) {
                        System.out.println(String.format("Found better classifier with threshold %f, accuracy = %3f", threshold, pctCorrect));
                    }
                }
            }
            m_inCV = false;

            System.out.printf(SUMMARY_STRING, bestThreshold, bestPercent);

            // Finally shrink back to the user specified threshold
            shrinkCentroids(m_shrinkageThreshold);
        }
    }

    private void shrinkCentroids(double thresh) {
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
                double dPrimeik = getDPrime(m_tStatisticsDik[i][k], thresh);
                //x(hat)i + mk(si + so)d'ik
                double newAttrValue = m_globalCentroid.getValue(i) + (thisMk * (m_withinClassStdDevSi[i] + m_medianSo) * dPrimeik);
                classCentroid.setShrunkenValue(i, newAttrValue);
            }
        }
    }

    private void calculateAllTStatistics() {
        // Make a 2d array to store the values with i rows (i attributes) and k columns (k classes)
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

                // Save the value
                m_tStatisticsDik[i][k] = dik;

                if (dik > m_maxTStatistic)
                    m_maxTStatistic = dik;
            }
        }
    }

    private double getDPrime(double dik, double shrinkageThresh) {
        // Equation 5 in the paper

        // Get the absolute difference between the value and delta
        double difference  = Math.abs(dik) - shrinkageThresh;
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

    private void calculateStandardDeviations(Instances trainingData) {
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

    private void calculateCentroids(Instances trainingData) {
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
        if (m_applyShrinkage) {
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
                    double squaredDif = classCentroid.getDifferenceFromInstanceAttribute(testInstance, i, true);
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
                "shrinks the centroids toward the global centroid, controlled by the threshold parameter.";
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
