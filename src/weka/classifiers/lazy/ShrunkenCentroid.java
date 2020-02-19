package weka.classifiers.lazy;

import org.jetbrains.annotations.NotNull;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.awt.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class ShrunkenCentroid extends AbstractClassifier {

    protected int m_K = 1;

    protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

    @OptionMetadata(displayName = "number of neighbours", description = "Number of neighbours to use (default = 1).",
            commandLineParamName = "K", commandLineParamSynopsis = "-K <int>", displayOrder = 1)
    public void setK(int k) {
        m_K = k;
    }

    public int getK() {
        return m_K;
    }

    private Centroid m_globalCentroid;

    private Centroid[] m_classCentroids;

    private int m_centroidNumAttributes;

    public void buildClassifier(Instances trainingData) throws Exception {
        trainingData = new Instances(trainingData);
        trainingData.deleteWithMissingClass();

        createCentroids(trainingData);

        // Calculate Si for each i (for each attribute)
        double[] withinClassStandardDeviations = calculateStandardDeviations(trainingData);
        double stdDevMedian = calculateMedian(withinClassStandardDeviations);
        Map<String, Double> allMK = calculateStandardizingParams(trainingData);
        
        m_NNSearch.setInstances(trainingData);
    }

    private Map<String, Double> calculateStandardizingParams(Instances trainingData) {
        // Calculate an mk for each class
        Set<String> classVals = m_classCentroids.keySet();
        Map<String, Double> allMK = new HashMap<>();

        double oneOverAll = 1 / trainingData.numInstances();

        for (String classVal : classVals) {
            double firstEq = 1 / m_classCentroids.get(classVal).getInstances().size();
            double mk = Math.sqrt(firstEq + oneOverAll);
            allMK.put(classVal, mk);
        }

        return allMK;
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
        int numClasses = m_classCentroids.length;
        double[] withinClassStandardDeviations = new double[m_centroidNumAttributes];
        // 1 / n - K
        double stdValue = 1 / (float) (trainingData.numInstances() - numClasses);

        for (int i = 0; i < trainingData.numAttributes(); i++) {
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
        for (int i = 0; i < m_classCentroids.length; i++) {
            Centroid c = m_classCentroids[i];
            double tmpDist = c.getDistanceFromInstance(testInstance);
            if (tmpDist < minDist) {
                minDist = tmpDist;
                minDistClass = i;
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
