package weka.classifiers.lazy;

import org.jetbrains.annotations.NotNull;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.util.Arrays;

public class ShrunkenCentroid extends AbstractClassifier {

    protected double m_delta = 1;

    private Centroid m_globalCentroid;

    private Centroid[] m_classCentroids;

    private int m_centroidNumAttributes;

    double[] allSi;
    // Calculate So (simply use the median of the Si values
    double soMedian;

    double[] allMK;

    double[][] tStatistics;

    public void buildClassifier(Instances trainingData) throws Exception {
        trainingData = new Instances(trainingData);
        trainingData.deleteWithMissingClass();

        createCentroids(trainingData);

        // Calculate Si for each i (for each attribute)
        allSi = calculateStandardDeviations(trainingData);
        // Calculate So (simply use the median of the Si values
        soMedian = calculateMedian(allSi);
        // Calculate Mk for all classes
        allMK = calculateStandardizingParamsForAllK(trainingData);

        // Calculate d'ik for all i and k
        calculateAllTStatistics(trainingData);

        // Shrink the centroids
        shrinkCentroids();
    }

    private void shrinkCentroids() {
        // Using the values calculated, we finally shrink the centroids
        // equation 4 in the paper
        for (int k = 0; k < m_classCentroids.length; k++) {

            // Get this class centroid and class Mk
            Centroid classCentroid = m_classCentroids[k];
            double thisMk = allMK[k];

            for (int i = 0; i < m_centroidNumAttributes; i++) {
                //x(hat)i + mk(si + so)d'ik
                double newAttrValue = m_globalCentroid.getValue(i) + thisMk * (allSi[i] + soMedian) * tStatistics[i][k];
                classCentroid.setValue(i, newAttrValue);
            }
        }
    }

    private void calculateAllTStatistics(Instances trainingData) {
        // Make a 2d array with i rows (i attributes) and k columns (k classes)
        // dik
        tStatistics = new double[m_centroidNumAttributes][m_classCentroids.length];

        // Now iterate over all attributes and calculate the t statistic for each class
        // (equation 1 in the paper, for calculating dik)
        for (int k = 0; k < m_classCentroids.length; k++) {

            // Get this class centroid and class Mk
            Centroid classCentroid = m_classCentroids[k];
            double thisMk = allMK[k];

            for (int i = 0; i < m_centroidNumAttributes; i++) {
                // Top half of equation
                double difFromGlobal = m_globalCentroid.getDifferenceFromInstanceAttribute(classCentroid.getInstance(), i);
                // Bottom half of equation
                double stdError = thisMk * (allSi[i] + soMedian);

                double dik = difFromGlobal / stdError;
                // Now calculate d'ik
                double dPrime = getDPrime(dik);
                // Save the value
                tStatistics[i][k] = dPrime;
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

        int sign = dik < 0 ? -1 : 1;
        difference *= sign;
        return difference;
    }

    private double[] calculateStandardizingParamsForAllK(Instances trainingData) {
        // Calculate an mk for each class
        double[] mkForAllK = new double[m_classCentroids.length];

        double oneOverAll = 1f / trainingData.numInstances();

        // Loop over all classes
        for (int i = 0; i < m_classCentroids.length; i++) {
            // Get the relevant class centroid
            Centroid c = m_classCentroids[i];
            // Calculate the first half of the equation (TODO give more meaningful names)
            double firstEq = 1f / c.getInstances().size();
            // Calculate the actual Mk
            double mk = Math.sqrt(firstEq + oneOverAll);
            // Save the value
            mkForAllK[i] = mk;
        }

        return mkForAllK;
    }

    private double calculateMedian(double[] values) {
        // Copy it as we need to sort the values to find median
        double[] valuesCopy = values.clone();
        Arrays.sort(valuesCopy);

        int middle = valuesCopy.length / 2;
        if (valuesCopy.length % 2 == 1) {
            return valuesCopy[middle];
        } else {
            return (valuesCopy[middle-1] + valuesCopy[middle]) / 2.0;
        }
    }

    private double[] calculateStandardDeviations(@NotNull Instances trainingData) {
        double[] withinClassStandardDeviations = new double[m_centroidNumAttributes];
        // 1 / n - K
        double stdValue = 1 / (float) (trainingData.numInstances() - m_classCentroids.length);

        // We want to calculate Si for all i
        for (int i = 0; i < m_centroidNumAttributes; i++) {
            double sum = 0;
            // Dont calculate for class attribute
            if (trainingData.attribute(i) != trainingData.classAttribute()) {
                // Outer sum : for all classes
                for (Centroid thisClassCentroid : m_classCentroids) {
                    // Inner sum : for all instances in this class
                    for (Instance instance : thisClassCentroid.getInstances()) {
                        // Xij - X-ik - difference between this instance and the class centroid for this attribute
                        double dif = thisClassCentroid.getDifferenceFromInstanceAttribute(instance, i);
                        // Square the value
                        dif = dif * dif;
                        // Add it to the sum for this attribute
                        sum += dif;
                    }
                }
                // Multiply it by the standard value
                sum *= stdValue;
                // Square root it
                sum = Math.sqrt(sum);
                // Save the standard deviation for this attribute
                withinClassStandardDeviations[i] = sum;
            }
        }
        return withinClassStandardDeviations;
    }

    private void initClassCentroids(Instances trainingData) {
        m_classCentroids = new Centroid[trainingData.classAttribute().numValues()];
        for (int i = 0; i < m_classCentroids.length; i++) {
            m_classCentroids[i] = new Centroid(m_centroidNumAttributes);
        }
    }

    private void createCentroids(@NotNull Instances trainingData) {
        m_centroidNumAttributes = trainingData.numAttributes() - 1; // ignore class value
        initClassCentroids(trainingData);
        m_globalCentroid = new Centroid(m_centroidNumAttributes);

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
        double minDist = Double.MAX_VALUE;
        double minDistClass = 0;
        // Here we use equation [6]
        for (int k = 0; k < m_classCentroids.length; k++) {
            Centroid c = m_classCentroids[k];
            double distanceSum = 0;
            // sum from i = 1 to p
            for (int i = 0; i < m_centroidNumAttributes; i++) {
                double squaredDif = c.getDifferenceFromInstanceAttribute(testInstance, i);
                // Actually square it
                squaredDif *= squaredDif;
                // bottom half of eq - (si + so)^2
                double standardizeVal = allSi[i] + soMedian;
                standardizeVal *= standardizeVal;
                // Add this value into the sum over all attributes
                double thisSum = squaredDif / standardizeVal;
                distanceSum += thisSum;
            }
            double classPrior = c.getInstances().size() / (float) m_globalCentroid.getInstances().size();
            double classPriorCorrection = distanceSum - (2 * Math.log(classPrior));
            if (classPriorCorrection < minDist) {
                minDist = classPriorCorrection;
                minDistClass = k;
            }
        }
        return minDistClass;
    }

//    public double[] distributionForInstance(Instance testInstance) throws Exception {
//
//        m_NNSearch.addInstanceInfo(testInstance);
//
//        Instances neighbours = m_NNSearch.kNearestNeighbours(testInstance, m_K);
//
//        double[] dist = new double[testInstance.numClasses()];
//        for (Instance neighbour : neighbours) {
//            if (testInstance.classAttribute().isNominal()) {
//                dist[(int)neighbour.classValue()] += 1.0 / neighbours.numInstances();
//            } else {
//                dist[0] += neighbour.classValue() / neighbours.numInstances();
//            }
//        }
//        return dist;
//    }
}
