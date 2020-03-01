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
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableClassifier;
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
 * <!-- globalinfo-end -->
 *
 * <!-- options-start -->
 * Valid options are: <p/>
 * <p>
 * <pre> -S &lt;shrinkage threshold&gt;
 *  Shrinkage threshold to use if no internal cross-validation is performed.
 *  (default 0.5) </pre>
 * <p/>
 * <p>
 * <pre> -num-thresholds &lt;number of thresholds&gt;
 *  Number of thresholds (fixed threshold will be used if this value is smaller than 2)
 *  (default 30) </pre>
 *  <p/>
 * <!-- options-end -->
 *
 * <!-- technical-bibtex-start -->
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
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Rhys Compton (rhys.compton@gmail.com)
 * @version $Revision$
 */

public class ShrunkenCentroid extends RandomizableClassifier {

    // We want this to make the classifier uniquely identifiable
    static final long serialVersionUID = 6254290359623814525L;

    /* Getters and setters for main hyperparameters */
    protected double m_shrinkageThreshold = 0.5;
    @OptionMetadata(
            displayName = "Fixed shrinkage threshold",
            description = "Shrinkage threshold to use if no internal cross-validation is performed.",
            displayOrder = 1,
            commandLineParamName = "shrinkage-threshold",
            commandLineParamSynopsis = "-shrinkage-threshold")
    public double getShrinkage() {
        return m_shrinkageThreshold;
    }
    public void setShrinkage(double d) {
        m_shrinkageThreshold = d;
    }

    protected int m_numEvaluationThresholds = 30;
    @OptionMetadata(
            displayName = "Number of thresholds (fixed threshold will be used if this value is smaller than 2)",
            description = "The number of thresholds to evaluate during internal CV.",
            displayOrder = 2,
            commandLineParamName = "num-thresholds",
            commandLineParamSynopsis = "-num-thresholds")
    public int getNumEvaluationThresholds() {
        return m_numEvaluationThresholds;
    }
    public void setNumEvaluationThresholds(int m_numEvaluationThresholds) {
        this.m_numEvaluationThresholds = m_numEvaluationThresholds;
    }

    // The number of folds for internal cross-validation
    protected int m_numFolds = 10;
    @OptionMetadata(
            displayName = "k for internal k-fold CV",
            description = "The number of folds for internal cross-validation.",
            displayOrder = 3,
            commandLineParamName = "num-folds",
            commandLineParamSynopsis = "-num-folds")
    public int getNumFolds() {
        return m_numFolds;
    }
    public void setNumFolds(int m_numFolds) {
        this.m_numFolds = m_numFolds;
    }

    // The per-class centroids
    protected double[][] m_centroids;

    // The global centroid
    protected double[] m_globalCentroid;

    // The threshold obtained from training;
    protected double m_bestThreshold;

    // The pooled within class standard deviations
    protected double[] m_standardDeviations;

    // Fudge factor
    protected double m_s_0;

    // Total number of training instances
    protected double m_numInstances;

    // Number of instances in each class
    protected double[] m_numInstancesInClass;

    // Header of training instances for output
    protected Instances m_header;

    private final String SUMMARY_STRING = "\nBest values found through CV:\nThreshold - %3f   |   Accuracy - %3f\n";

    /**
     * Method used for building the classifier.
     * @param trainingData the data used for training
     * @throws Exception if construction fails
     */
    public void buildClassifier(Instances trainingData) throws Exception {

        // Can classifier handle the data?
        getCapabilities().testWithFail(trainingData);

        // Keep only data with known class values
        trainingData = new Instances(trainingData);
        trainingData.deleteWithMissingClass();

        // Calculate the various statistics required
        m_centroids = new double[trainingData.numClasses()][trainingData.numAttributes()];
        m_numInstancesInClass = new double[trainingData.numClasses()];
        m_numInstances = 0;
        m_globalCentroid = new double[trainingData.numAttributes()];
        for (Instance instance : trainingData) {
            int classVal = (int) instance.classValue();
            m_numInstancesInClass[classVal]++;
            for (int i = 0; i < instance.numAttributes(); i++) {
                m_centroids[classVal][i] += instance.value(i);
            }
        }
        for (int j = 0; j < trainingData.numClasses(); j++) {
            for (int i = 0; i < trainingData.numAttributes(); i++) {
                m_globalCentroid[i] += m_centroids[j][i];
            }
            m_numInstances += m_numInstancesInClass[j];
        }
        for (int j = 0; j < trainingData.numClasses(); j++) {
            for (int i = 0; i < trainingData.numAttributes(); i++) {
                m_centroids[j][i] /= m_numInstancesInClass[j];
            }
        }
        for (int i = 0; i < trainingData.numAttributes(); i++) {
            m_globalCentroid[i] /= m_numInstances;
        }
        m_standardDeviations = new double[trainingData.numAttributes()];
        for (Instance instance : trainingData) {
            int classVal = (int) instance.classValue();
            for (int i = 0; i < instance.numAttributes(); i++) {
                double diff = m_centroids[classVal][i] - instance.value(i);
                m_standardDeviations[i] += diff * diff;
            }
        }
        for (int i = 0; i < trainingData.numAttributes(); i++) {
            m_standardDeviations[i] /= (m_numInstances - trainingData.numClasses());
            m_standardDeviations[i] = Math.sqrt(m_standardDeviations[i]);
        }

        // Make sure class attribute is treated correctly in subsequent code
        m_standardDeviations[trainingData.classIndex()] = 0;
        for (int j = 0; j < trainingData.numClasses(); j++) {
            m_centroids[j][trainingData.classIndex()] = 0;
        }
        m_globalCentroid[trainingData.classIndex()] = 0;
        m_s_0 = Utils.kthSmallestValue(m_standardDeviations, 1 + trainingData.numAttributes() / 2);
        if ((trainingData.numAttributes() - 1) % 2 == 0) { // Even number of predictor attributes?
            m_s_0 = (m_s_0 + Utils.kthSmallestValue(m_standardDeviations,
                    2 + trainingData.numAttributes() / 2)) / 2.0;
        }

        // Perform internal cross-validation to determine best shrinkage threshold
        m_bestThreshold = 0;
        if (getNumEvaluationThresholds() > 1) {

            // Set up template classifier, evaluation objects, and data for stratified cross-validation
            ShrunkenCentroid sC = new ShrunkenCentroid();
            sC.setNumEvaluationThresholds(-1);
            Evaluation[] scores = new Evaluation[getNumEvaluationThresholds()];
            for (int j = 0; j < getNumEvaluationThresholds(); j++) {
                scores[j] = new Evaluation(trainingData);
            }

            // Figure out the thresholds to consider
            double maxThreshold = -1;
            for (int j = 0; j < trainingData.numClasses(); j++) {
                double m_j = Math.sqrt((1.0/m_numInstancesInClass[j]) - (1.0/m_numInstances)); // Bug in paper?
                for (int i = 0; i < trainingData.numAttributes(); i++) {
                    double d_ij = m_centroids[j][i] - m_globalCentroid[i];
                    double denom = m_j * (m_standardDeviations[i] + m_s_0);
                    maxThreshold = Math.max(Math.abs(d_ij / denom), maxThreshold);
                }
            }
            double inc = maxThreshold / (getNumEvaluationThresholds() - 1);

            // Run k-fold CV for all thresholds
            Instances data = new Instances(trainingData);
            Random random = new Random(getSeed());
            data.randomize(random);
            data.stratify(getNumFolds());
            for (int j = 0; j < getNumFolds(); j++) {
                Instances train = data.trainCV(getNumFolds(), j, random);
                Instances test = data.testCV(getNumFolds(), j);
                double threshold = 0;
                Classifier copiedClassifier = AbstractClassifier.makeCopy(sC);
                copiedClassifier.buildClassifier(train);
                for (int i = 0; i < getNumEvaluationThresholds(); i++) {
                    ((ShrunkenCentroid) copiedClassifier).m_bestThreshold = threshold;
                    scores[i].setPriors(train);
                    scores[i].evaluateModel(copiedClassifier, test);
                    threshold += inc;
                }
            }

            // Establish best threshold based on results from k-fold CV
            double bestPercent = -1;
            double threshold = 0;
            for (int i = 0; i < getNumEvaluationThresholds(); i++) {
                double pctCorrect = scores[i].pctCorrect();
                if (m_Debug) {
                    System.err.printf("%3f --- %3f\n", threshold, pctCorrect);
                }
                if (pctCorrect > bestPercent) {
                    m_bestThreshold = threshold;
                    bestPercent = pctCorrect;
                    if (m_Debug) {
                        System.err.println(String.format("Found better classifier with threshold %f, accuracy = %3f",
                                threshold, pctCorrect));
                    }
                }
                threshold += inc;
            }
            if (m_Debug) {
                System.err.printf(SUMMARY_STRING, m_bestThreshold, bestPercent);
            }
        } else {
            m_bestThreshold = getShrinkage();
        }
        m_header = trainingData.stringFreeStructure();
    }

    /**
     * Returns the estimated class probability distribution for the given instance.
     *
     * @param testInstance the instance to obtain class probabilities for
     * @return the class probability estimates as an array of floating-point numbers
     */
    public double[] distributionForInstance(Instance testInstance) {

        double[] dist = new double[m_centroids.length];
        for (int k = 0; k < m_centroids.length; k++) {
            double m_k = Math.sqrt((1.0 / m_numInstancesInClass[k]) - (1.0 / m_numInstances)); // Bug in paper
            for (int i = 0; i < m_centroids[k].length; i++) {
                if (i != testInstance.classIndex()) {
                    double d_ij = m_centroids[k][i] - m_globalCentroid[i]; // 0 for class attribute
                    double denom = m_k * (m_standardDeviations[i] + m_s_0);
                    double adjustWithoutSign = Math.abs(d_ij / denom) - m_bestThreshold;
                    double diff = testInstance.value(i);
                    if (adjustWithoutSign > 0) {
                        if (d_ij <= 0) {
                            diff -= (m_globalCentroid[i] - denom * adjustWithoutSign);
                        } else {
                            diff -= (m_globalCentroid[i] + denom * adjustWithoutSign);
                        }
                    } else {
                        diff -= m_globalCentroid[i];
                    }
                    double sqrtDenom = m_standardDeviations[i] + m_s_0;
                    dist[k] += (diff * diff) / (sqrtDenom * sqrtDenom);
                }
            }
            dist[k] -= 2 * Math.log(m_numInstancesInClass[k] / m_numInstances);
            dist[k] *= -0.5;
        }
        return Utils.logs2probs(dist);
    }

    /**
     * Returns a textual description of the classifier.
     */
    public String toString() {

        if (m_header == null) {
            return "Classifier has not been built yet.";
        }
        StringBuffer result = new StringBuffer();
        result.append("\n=== Shrunken centroid classifier with threshold " + m_bestThreshold + " ===");
        for (int k = 0; k < m_centroids.length; k++) {
            result.append("\n\nShrunken centroid for class  " + m_header.classAttribute().value(k) + "\n\n");
            double m_k = Math.sqrt((1.0 / m_numInstancesInClass[k]) - (1.0 / m_numInstances)); // Bug in paper
            for (int i = 0; i < m_centroids[k].length; i++) {
                if (i != m_header.classIndex()) {
                    double d_ij = m_centroids[k][i] - m_globalCentroid[i]; // 0 for class attribute
                    double denom = m_k * (m_standardDeviations[i] + m_s_0);
                    double adjustWithoutSign = Math.abs(d_ij / denom) - m_bestThreshold;
                    if (adjustWithoutSign > 0) {
                        if (d_ij <= 0) {
                            result.append(m_header.attribute(i).name() + ": " +
                                    (m_globalCentroid[i] - denom * adjustWithoutSign) + "\n");
                        } else {
                            result.append(m_header.attribute(i).name() + ": " +
                                    (m_globalCentroid[i] + denom * adjustWithoutSign) + "\n");
                        }
                    }
                }
            }
        }
        return result.toString();
    }

    /**
     * The info shown in the GUI.
     * @return the info describing the filter.
     */
    public String globalInfo() {
        return "This algorithm performs shrunken nearest-centroid classification.";
    }

    /**
     * The capabilities of this classifier.
     * @return the capabilities
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        return result;
    }

    /**
     * Main method for testing this class
     *
     * @param argv options
     */
    public static void main(String [] argv){
        runClassifier(new ShrunkenCentroid(), argv);
    }
}